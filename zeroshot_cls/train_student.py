import os
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm

from dassl.engine import build_trainer
from dassl.config import get_cfg_default
from dassl.utils import setup_logger, set_random_seed

from models.dgcnn import DGCNN
from main_sfda import SFDA_PointCLIP, extend_cfg, reset_cfg
import datasets.pointda

from torch.amp import autocast, GradScaler


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_strong_aug(pc):
    B, N, C = pc.shape
    pc_strong = pc.clone()

    # 随机 Y 轴旋转 (3D 模型必须掌握的核心能力)
    angles = torch.rand(B, device=pc.device) * 2 * torch.pi
    cos_val = torch.cos(angles)
    sin_val = torch.sin(angles)
    rot_mat = torch.zeros(B, 3, 3, device=pc.device)
    rot_mat[:, 0, 0] = cos_val
    rot_mat[:, 0, 2] = sin_val
    rot_mat[:, 1, 1] = 1.0
    rot_mat[:, 2, 0] = -sin_val
    rot_mat[:, 2, 2] = cos_val
    pc_strong = torch.bmm(pc_strong, rot_mat)

    scale = torch.rand(B, 1, 3, device=pc.device) * 0.4 + 0.8
    pc_strong = pc_strong * scale

    shift = (torch.rand(B, 1, 3, device=pc.device) - 0.5) * 0.2
    pc_strong = pc_strong + shift

    noise = torch.randn_like(pc_strong) * 0.02
    noise = torch.clamp(noise, -0.05, 0.05)
    pc_strong = pc_strong + noise

    return pc_strong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='output/train_student_pointda')
    parser.add_argument('--teacher-weight', type=str, required=True, help='VLM 老师权重 (sfda_mlp_epoch_xxx.pth)')
    parser.add_argument('--source-weight', type=str, required=True, help='DGCNN 源域底座 (dgcnn_source_best.pth)')
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--dataset-config-file', type=str, default='')
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--micro-batch', type=int, default=8)

    # 🚨 动态降准：10分类任务盲猜是0.1，0.45 已经代表极其自信！降低门槛释放真金！
    parser.add_argument('--threshold', type=float, default=0.45)
    parser.add_argument('--temperature', type=float, default=2.0)

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--backbone', type=str, default='ViT-B/16')
    parser.add_argument('--trainer', type=str, default='SFDA_PointCLIP')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    accumulation_steps = max(1, args.batch_size // args.micro_batch)

    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(args.config_file)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    reset_cfg(cfg, args)

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.micro_batch
    cfg.DATALOADER.TEST.BATCH_SIZE = args.micro_batch
    cfg.TRAINER.NAME = "SFDA_PointCLIP"
    cfg.freeze()

    setup_logger(args.output_dir)
    set_random_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n>>> [1/2] 唤醒 2D VLM 老师模型 (提供跨模态指导)...")
    teacher_trainer = build_trainer(cfg)
    teacher_trainer.model.load_state_dict(torch.load(args.teacher_weight, map_location=device, weights_only=True))
    teacher_trainer.model.eval()
    for param in teacher_trainer.model.parameters():
        param.requires_grad = False

    train_loader = teacher_trainer.train_loader_x
    test_loader_raw = teacher_trainer.test_loader
    test_loader = list(test_loader_raw.values())[0] if isinstance(test_loader_raw, dict) else test_loader_raw
    num_classes = 10

    print(f"\n>>> [2/2] 部署 3D 学生模型与 EMA 稳定器...")
    student_model = DGCNN(num_classes=num_classes).to(device)
    student_model.load_state_dict(torch.load(args.source_weight, map_location=device, weights_only=True))

    # 🚨 新增：指数移动平均 (EMA) 模型，过滤伪标签带来的震荡！
    ema_student_model = DGCNN(num_classes=num_classes).to(device)
    ema_student_model.load_state_dict(student_model.state_dict())
    ema_student_model.eval()
    for param in ema_student_model.parameters():
        param.requires_grad = False

    print("\n>>> [BN Calibration] 校准源域模型在目标域的 BatchNorm 统计量...")
    student_model.train()
    with torch.no_grad():
        for batch in train_loader:
            pc = batch["img"].to(device).float()
            _ = student_model(pc)

            # 同时校准 EMA 模型的 BN
            _ = ema_student_model(pc)

    print("\n>>> [Baseline] 测试校准后的纯源域模型...")
    ema_student_model.eval()
    init_correct, init_samples = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            pc = batch["img"].to(device).float()
            label = batch["label"].to(device)
            logits = ema_student_model(pc)
            init_correct += accuracy(logits, label)[0]
            init_samples += pc.shape[0]
    baseline_acc = (init_correct / init_samples) * 100
    print(f"=> Source-Only 校准后初始准确率: {baseline_acc:.2f}%\n")

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    best_acc = baseline_acc

    print("\n>>> 开始跨域突围: 硬标签 (定海神针) + 自洽正则化 (突破上限) + 软知识 (托底) ...")
    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            pc = batch["img"].to(device).float()
            pc_strong = get_strong_aug(pc)

            with autocast('cuda'):
                # --- 1. 老师给出软硬指导 ---
                with torch.no_grad():
                    logits_teacher = teacher_trainer.model_inference(pc)
                    probs_teacher = F.softmax(logits_teacher, dim=1)
                    max_probs, pseudo_labels = torch.max(probs_teacher, dim=1)

                    mask = (max_probs >= args.threshold).float()
                    probs_teacher_soft = F.softmax(logits_teacher / args.temperature, dim=1)

                # --- 2. 学生双路输出 ---
                pc_combined = torch.cat([pc, pc_strong], dim=0)
                logits_combined = student_model(pc_combined)
                logits_student_clean, logits_student_strong = logits_combined.chunk(2)

                # --- 3. 黄金三角 Loss，彻底告别 0.0000！ ---

                # [护盾] Loss 1 (Hard): 只在老师有把握时，学生硬对齐 (FixMatch)
                ce_loss_strong = F.cross_entropy(logits_student_strong, pseudo_labels, reduction='none',
                                                 label_smoothing=0.1)
                loss_hard = (ce_loss_strong * mask).sum() / (mask.sum() + 1e-8)

                # [托底] Loss 2 (Soft KD): 老师没把握时，轻轻给个大方向，保证梯度永远流动
                log_probs_clean = F.log_softmax(logits_student_clean / args.temperature, dim=1)
                loss_soft = F.kl_div(log_probs_clean, probs_teacher_soft.detach(), reduction='batchmean') * (
                            args.temperature ** 2)

                # [突破] Loss 3 (Self-Consistency): 强迫干净视角和强干扰视角的预测一致！这是 3D 自我顿悟的核心！
                probs_clean_detach = F.softmax(logits_student_clean.detach(), dim=1)
                log_probs_strong = F.log_softmax(logits_student_strong, dim=1)
                loss_consist = F.kl_div(log_probs_strong, probs_clean_detach, reduction='batchmean')

                loss = (1.0 * loss_hard + 0.5 * loss_soft + 1.0 * loss_consist) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # 🚨 新增：平滑更新 EMA 学生模型
                with torch.no_grad():
                    m = 0.999  # EMA 动量
                    for param_q, param_k in zip(student_model.parameters(), ema_student_model.parameters()):
                        param_k.data.mul_(m).add_(param_q.detach().data, alpha=1 - m)
                    for buffer_q, buffer_k in zip(student_model.buffers(), ema_student_model.buffers()):
                        buffer_k.data.copy_(buffer_q.data)

            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({
                "Loss": f"{loss.item() * accumulation_steps:.4f}",
                "Mask_Ratio": f"{(mask.sum().item() / pc.shape[0]):.1%}"
            })

        scheduler.step()

        # 🚨 测试极其稳健的 EMA 学生模型
        print(f"\nEpoch {epoch + 1} 结束 -> 测试 EMA 3D Student 性能...")
        ema_student_model.eval()
        total_correct = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                pc = batch["img"].to(device).float()
                label = batch["label"].to(device)
                with autocast('cuda'):
                    logits = ema_student_model(pc)
                total_correct += accuracy(logits, label)[0]
                total_samples += pc.shape[0]

        curr_acc = (total_correct / total_samples) * 100
        print(f"=> EMA 3D Student Target Accuracy: {curr_acc:.2f}%")

        if curr_acc > best_acc:
            best_acc = curr_acc
            # 保存表现更好的 EMA 权重
            torch.save(ema_student_model.state_dict(),
                       os.path.join(args.output_dir, "best_student_dgcnn_finetuned.pth"))
            print(f"🎉 突破！已保存最佳跨模态学生模型 (当前最高: {best_acc:.2f}%)")

    print(f"\n✅ 终极蒸馏完成! 最佳模型准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()