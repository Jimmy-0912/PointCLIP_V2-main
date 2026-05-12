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
    """
    🚨 恢复安全的 3D 强增强，彻底移除会破坏 DGCNN K-NN 图结构的 Point Dropout！
    """
    B, N, C = pc.shape
    pc_strong = pc.clone()
    device = pc.device

    # 1. 随机 Y 轴旋转 (打破视角依赖，最核心的增强)
    angles = torch.rand(B, device=device) * 2 * torch.pi
    cos_val = torch.cos(angles)
    sin_val = torch.sin(angles)
    rot_mat = torch.zeros(B, 3, 3, device=device)
    rot_mat[:, 0, 0] = cos_val
    rot_mat[:, 0, 2] = sin_val
    rot_mat[:, 1, 1] = 1.0
    rot_mat[:, 2, 0] = -sin_val
    rot_mat[:, 2, 2] = cos_val
    pc_strong = torch.bmm(pc_strong, rot_mat)

    # 2. 随机缩放与平移
    scale = torch.rand(B, 1, 3, device=device) * 0.4 + 0.8
    pc_strong = pc_strong * scale
    shift = (torch.rand(B, 1, 3, device=device) - 0.5) * 0.2
    pc_strong = pc_strong + shift

    # 3. 局部噪声注入
    noise = torch.randn_like(pc_strong) * 0.02
    noise = torch.clamp(noise, -0.05, 0.05)
    pc_strong = pc_strong + noise

    return pc_strong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='output/train_student_pointda')
    parser.add_argument('--teacher-weight', type=str, required=True, help='VLM 老师权重')
    parser.add_argument('--source-weight', type=str, required=True, help='DGCNN 源域底座')
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--dataset-config-file', type=str, default='')
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--micro-batch', type=int, default=8)

    # 🚨 调回 0.55 提取高质量真金，T=2.0 软化分布
    parser.add_argument('--threshold', type=float, default=0.55)
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

    print(f"\n>>> [1/2] 唤醒 2D VLM 老师模型 (全程提供语义锚点)...")
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

    print("\n>>> 开始黄金三角蒸馏: VLM 硬标签 + VLM 软知识 + EMA 自洽稳定 ...")
    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            pc = batch["img"].to(device).float()
            pc_strong = get_strong_aug(pc)

            with autocast('cuda'):
                with torch.no_grad():
                    # --- 1. 获取 VLM 老师的语义知识 ---
                    logits_vlm = teacher_trainer.model_inference(pc)
                    probs_vlm = F.softmax(logits_vlm, dim=1)
                    max_probs_vlm, pseudo_labels = torch.max(probs_vlm, dim=1)

                    # 掩码与软分布
                    mask = (max_probs_vlm >= args.threshold).float()
                    probs_vlm_soft = F.softmax(logits_vlm / args.temperature, dim=1)

                    # --- 2. 获取 EMA 学生的稳健 3D 知识 ---
                    logits_ema_clean = ema_student_model(pc)
                    probs_ema_clean = F.softmax(logits_ema_clean, dim=1)

                # --- 3. 学生网络双路输入 ---
                pc_combined = torch.cat([pc, pc_strong], dim=0)
                logits_combined = student_model(pc_combined)
                logits_student_clean, logits_student_strong = logits_combined.chunk(2)

                # --- 🚨 黄金三角 Loss ---

                # [突破] Loss 1 (Hard FixMatch): 在老师有把握的样本上，强迫学生在强干扰下认出 VLM 的硬标签
                ce_loss_strong = F.cross_entropy(logits_student_strong, pseudo_labels, reduction='none',
                                                 label_smoothing=0.1)
                loss_hard = (ce_loss_strong * mask).sum() / (mask.sum() + 1e-8)

                # [保底] Loss 2 (Soft KD): 让干净点云去拟合 VLM 的软分布，全量数据托底，确保大方向与文本语义对齐
                log_probs_clean = F.log_softmax(logits_student_clean / args.temperature, dim=1)
                loss_soft = F.kl_div(log_probs_clean, probs_vlm_soft.detach(), reduction='batchmean') * (
                            args.temperature ** 2)

                # [鲁棒] Loss 3 (EMA Consistency): 用 EMA 提取的纯 3D 特征，去约束强干扰下的预测，补足 VLM 缺乏的三维抗干扰能力
                log_probs_strong = F.log_softmax(logits_student_strong, dim=1)
                loss_consist = F.kl_div(log_probs_strong, probs_ema_clean.detach(), reduction='batchmean')

                # 经典组合比例
                loss = (1.0 * loss_hard + 0.5 * loss_soft + 1.0 * loss_consist) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # 平滑更新 EMA 模型
                with torch.no_grad():
                    m = 0.999
                    for param_q, param_k in zip(student_model.parameters(), ema_student_model.parameters()):
                        param_k.data.mul_(m).add_(param_q.detach().data, alpha=1 - m)
                    for buffer_q, buffer_k in zip(student_model.buffers(), ema_student_model.buffers()):
                        buffer_k.data.copy_(buffer_q.data)

            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({
                "Loss": f"{loss.item() * accumulation_steps:.4f}",
                "Mask": f"{(mask.sum().item() / pc.shape[0]):.1%}"
            })

        scheduler.step()

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
        print(f"=> EMA Student Target Accuracy: {curr_acc:.2f}%")

        if curr_acc > best_acc:
            best_acc = curr_acc
            torch.save(ema_student_model.state_dict(),
                       os.path.join(args.output_dir, "best_student_dgcnn_finetuned.pth"))
            print(f"🎉 突破！已保存最佳模型 (当前最高: {best_acc:.2f}%)")

    print(f"\n✅ 终极跨域突围完成! 最佳模型准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()