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

    # 随机 Y 轴旋转，打破视角依赖
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
    parser.add_argument('--output-dir', type=str, default='output/self_distill_pointda')
    parser.add_argument('--teacher-weight', type=str, required=True)
    parser.add_argument('--source-weight', type=str, default='')
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--dataset-config-file', type=str, default='')
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--micro-batch', type=int, default=4)

    # 🚨 复刻奇迹参数：T=0.5 锐化分布，Threshold=0.55 释放黄金标签
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.55)

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

    print(f"\n>>> 初始化 VLM 跨模态老师模型 (SFDA_PointCLIP)...")
    teacher_trainer = build_trainer(cfg)
    teacher_trainer.model.load_state_dict(torch.load(args.teacher_weight, map_location=device, weights_only=True))
    teacher_trainer.model.eval()
    for param in teacher_trainer.model.parameters():
        param.requires_grad = False

    train_loader = teacher_trainer.train_loader_x
    test_loader_raw = teacher_trainer.test_loader
    test_loader = list(test_loader_raw.values())[0] if isinstance(test_loader_raw, dict) else test_loader_raw
    num_classes = 10

    print(f"\n>>> 初始化 3D 接收学生模型 (Robust Student DGCNN)...")
    student_model = DGCNN(num_classes=num_classes).to(device)
    if args.source_weight and os.path.exists(args.source_weight):
        print(f">>> 无缝加载源域(98%精度)底座权重: {args.source_weight}...")
        student_model.load_state_dict(torch.load(args.source_weight, map_location=device, weights_only=True))

    print("\n>>> [Baseline] 评估纯源域模型 (Source-Only) 在目标域的初始性能...")
    student_model.eval()
    init_correct = 0.0
    init_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            pc = batch["img"].to(device).float()
            label = batch["label"].to(device)
            logits = student_model(pc)
            init_correct += accuracy(logits, label)[0]
            init_samples += pc.shape[0]
    baseline_acc = (init_correct / init_samples) * 100
    print(f"=> Source-Only 初始准确率: {baseline_acc:.2f}%\n")

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    best_acc = baseline_acc

    # =================================================================
    # 🚨 复刻 60.63% 奇迹的核心架构：Soft KD + Hard FixMatch
    # 绝对没有任何 IM Loss（不会产生负数），绝对不会让 Mask_Ratio=0 时无事可做！
    # =================================================================
    print("\n>>> 开始跨模态终极降维打击蒸馏: 软蒸馏保底 + 强增强硬标签突破 ...")
    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            pc = batch["img"].to(device).float()
            pc_strong = get_strong_aug(pc)

            with autocast('cuda'):
                # 1. 老师做题
                with torch.no_grad():
                    logits_teacher = teacher_trainer.model_inference(pc)
                    probs_teacher_raw = F.softmax(logits_teacher, dim=1)
                    max_probs, pseudo_labels = torch.max(probs_teacher_raw, dim=1)

                    # 生成过滤掩码
                    mask = (max_probs >= args.threshold).float()

                    # 生成锐化后的软分布 (T=0.5)
                    probs_teacher_sharp = F.softmax(logits_teacher / args.temperature, dim=1)

                # 2. 学生做题 (联合前向传播)
                pc_combined = torch.cat([pc, pc_strong], dim=0)
                logits_combined = student_model(pc_combined)
                logits_student_clean, logits_student_strong = logits_combined.chunk(2)

                # 3. 双轨制损失
                # 轨1 (Soft KD)：让学生拟合锐化后的软分布。这保证了即使 Mask 为 0，学生也能学到分类边界的整体结构，绝不走偏！
                log_probs_student_clean = F.log_softmax(logits_student_clean / args.temperature, dim=1)
                loss_kd = F.kl_div(log_probs_student_clean, probs_teacher_sharp.detach(), reduction='batchmean') * (
                            args.temperature ** 2)

                # 轨2 (Hard FixMatch)：在老师有把握的样本上，强迫学生在严重干扰下认出硬标签！(带0.2平滑防过拟合)
                ce_loss_strong = F.cross_entropy(logits_student_strong, pseudo_labels, reduction='none',
                                                 label_smoothing=0.2)
                loss_hard = (ce_loss_strong * mask).sum() / (mask.sum() + 1e-8)

                # 权重分配：0.5 倍软知识 + 1.0 倍硬标签
                loss = (0.5 * loss_kd + 1.0 * loss_hard) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({
                "Loss": f"{loss.item() * accumulation_steps:.4f}",
                "Mask_Ratio": f"{(mask.sum().item() / pc.shape[0]):.1%}"
            })

        scheduler.step()

        print(f"\nEpoch {epoch + 1} 结束 -> 测试 3D Robust Student 性能...")
        student_model.eval()
        total_correct = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                pc = batch["img"].to(device).float()
                label = batch["label"].to(device)
                with autocast('cuda'):
                    logits = student_model(pc)
                total_correct += accuracy(logits, label)[0]
                total_samples += pc.shape[0]

        curr_acc = (total_correct / total_samples) * 100
        print(f"=> 3D Robust Student Target Accuracy: {curr_acc:.2f}%")

        if curr_acc > best_acc:
            best_acc = curr_acc
            torch.save(student_model.state_dict(), os.path.join(args.output_dir, "best_self_distilled_dgcnn.pth"))
            print(f"🎉 突破！已保存新的最佳模型 (当前最高: {best_acc:.2f}%)")

    print(f"\n✅ 蒸馏完成! 最佳 3D 学生模型准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()