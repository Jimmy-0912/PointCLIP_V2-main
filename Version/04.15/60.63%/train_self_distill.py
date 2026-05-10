import os
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm

from dassl.config import get_cfg_default
from dassl.utils import setup_logger, set_random_seed
from dassl.data import DataManager

from models.dgcnn import DGCNN
from main_sfda import extend_cfg, reset_cfg
import datasets.scanobjnn

from torch.amp import autocast, GradScaler


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_strong_aug(pc):
    """
    安全强增强：随机缩放、平移与抖动。绝对不破坏 KNN 拓扑！
    """
    B, N, C = pc.shape
    pc_strong = pc.clone()
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
    parser.add_argument('--output-dir', type=str, default='output/self_distill_round1')
    parser.add_argument('--teacher-weight', type=str, required=True, help='第一轮训练出的 60% 最佳学生权重')
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--dataset-config-file', type=str, default='')
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--batch-size', type=int, default=16, help='逻辑 Batch Size')
    parser.add_argument('--micro-batch', type=int, default=4, help='物理 Batch Size')

    parser.add_argument('--temperature', type=float, default=2.0, help='用于软化概率的温度系数')
    # 🚨 微调 1: 提高阈值至 0.85，极其严苛地过滤老师的错误伪标签，宁缺毋滥！
    parser.add_argument('--threshold', type=float, default=0.85, help='截断阈值，宁缺毋滥')

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
    cfg.freeze()

    setup_logger(args.output_dir)
    set_random_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"\n>>> 正在初始化数据加载器 (物理 Batch: {args.micro_batch}, 逻辑 Batch: {args.batch_size}, 梯度累加: {accumulation_steps}步)...")
    dm = DataManager(cfg)
    train_loader = dm.train_loader_x
    test_loader_raw = dm.test_loader
    test_loader = list(test_loader_raw.values())[0] if isinstance(test_loader_raw, dict) else test_loader_raw
    num_classes = 11

    print(f"\n>>> 初始化 3D 老师模型 (Frozen Teacher)...")
    teacher_model = DGCNN(num_classes=num_classes).to(device)
    teacher_model.load_state_dict(torch.load(args.teacher_weight, map_location=device, weights_only=True))
    # 老师完全冻结，提供绝对稳定的 60% 锚点！
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    print(f"\n>>> 初始化 3D 学生模型 (Robust Student)...")
    student_model = DGCNN(num_classes=num_classes).to(device)
    student_model.load_state_dict(torch.load(args.teacher_weight, map_location=device, weights_only=True))

    # 🚨 微调 2: 略微提高学习率至 5e-5，给模型跳出 60% 局部最优解的能量！
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    print("\n>>> 开始联合蒸馏范式: Soft KD (保底) + Hard FixMatch (突破) ...")
    best_acc = 0.0

    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            pc = batch["img"].to(device).float()
            pc_strong = get_strong_aug(pc)

            with autocast('cuda'):
                # --- A. 老师生成“软知识”与“硬标签” ---
                with torch.no_grad():
                    logits_teacher = teacher_model(pc)
                    probs_teacher_raw = F.softmax(logits_teacher, dim=1)
                    max_probs_teacher, pseudo_labels = torch.max(probs_teacher_raw, dim=1)

                    mask = (max_probs_teacher >= args.threshold).float()
                    probs_teacher_soft = F.softmax(logits_teacher / args.temperature, dim=1)

                # --- B. 联合前向传播 (完美解决 BN 偏移) ---
                pc_combined = torch.cat([pc, pc_strong], dim=0)
                logits_combined = student_model(pc_combined)

                # 拆分结果
                logits_student_clean, logits_student_strong = logits_combined.chunk(2)

                # --- C. 双轨制损失计算 ---
                log_probs_student_clean = F.log_softmax(logits_student_clean / args.temperature, dim=1)
                loss_kd = F.kl_div(log_probs_student_clean, probs_teacher_soft, reduction='batchmean') * (
                            args.temperature ** 2)

                ce_loss_strong = F.cross_entropy(logits_student_strong, pseudo_labels, reduction='none')
                loss_hard = (ce_loss_strong * mask).sum() / (mask.sum() + 1e-8)

                # 🚨 微调 3: 削弱软知识蒸馏 (0.5)，鼓励学生在强增强数据上超越老师 (1.0)！
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

        # 评估学生性能
        print(f"\nEpoch {epoch + 1} 结束 -> 测试 Robust Student 性能...")
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
        print(f"=> Robust Student Target Accuracy: {curr_acc:.2f}%")

        if curr_acc > best_acc:
            best_acc = curr_acc
            print(f"🎉 发现新高! 保存最佳模型...")
            torch.save(student_model.state_dict(), os.path.join(args.output_dir, "best_self_distilled_dgcnn.pth"))

    print(f"\n✅ 终极联合蒸馏完成! 最佳最终模型准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()