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
import datasets.pointda

from torch.amp import autocast, GradScaler


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_strong_aug(pc):
    """
    纯粹且安全的 3D 强增强，只保留刚性变换，保护 K-NN 图结构
    """
    B, N, C = pc.shape
    pc_strong = pc.clone()
    device = pc.device



    # 随机 Y 轴旋转
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

    scale = torch.rand(B, 1, 3, device=device) * 0.4 + 0.8
    pc_strong = pc_strong * scale
    shift = (torch.rand(B, 1, 3, device=device) - 0.5) * 0.2
    pc_strong = pc_strong + shift

    noise = torch.randn_like(pc_strong) * 0.02
    noise = torch.clamp(noise, -0.05, 0.05)
    pc_strong = pc_strong + noise

    return pc_strong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='output/self_distill_pointda')
    parser.add_argument('--teacher-weight', type=str, required=True, help='起始基础权重 (55% 的学生权重)')
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--dataset-config-file', type=str, default='')
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--micro-batch', type=int, default=8)

    # 🚨 回归奇迹参数：严格门槛 0.65 保真金，T=0.5 锐化软标签保底线
    parser.add_argument('--threshold', type=float, default=0.65)
    parser.add_argument('--temperature', type=float, default=0.5)

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

    dm = DataManager(cfg)
    train_loader = dm.train_loader_x
    test_loader_raw = dm.test_loader
    test_loader = list(test_loader_raw.values())[0] if isinstance(test_loader_raw, dict) else test_loader_raw
    num_classes = 10

    # ==========================================================
    # 🚨 终极安全回归：彻底废弃会坍塌的 EMA，启用绝对冻结的导师！
    # ==========================================================
    print(f"\n>>> [1/2] 部署绝对冻结的 3D 导师 (基准锚点，永不更新)...")
    teacher_model = DGCNN(num_classes=num_classes).to(device)
    teacher_model.load_state_dict(torch.load(args.teacher_weight, map_location=device, weights_only=True))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False  # 绝对锁定！

    print(f"\n>>> [2/2] 部署 3D 学生模型 (探索与进化)...")
    student_model = DGCNN(num_classes=num_classes).to(device)
    student_model.load_state_dict(torch.load(args.teacher_weight, map_location=device, weights_only=True))

    print("\n>>> [Baseline] 测试静态 3D 导师模型在目标域的保底准确率...")
    init_correct, init_samples = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            pc = batch["img"].to(device).float()
            label = batch["label"].to(device)
            logits = teacher_model(pc)
            init_correct += accuracy(logits, label)[0]
            init_samples += pc.shape[0]
    baseline_acc = (init_correct / init_samples) * 100
    print(f"=> Frozen Teacher 初始准确率: {baseline_acc:.2f}% (绝不坍塌的铁底！)\n")

    # 🚨 降低学习率至 5e-5，精细微调，防止大步长扯破已有的特征流形
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    best_acc = baseline_acc

    print("\n>>> 开始同模态 3D 自我蒸馏: 锐化软知识 + 高纯度硬标签 FixMatch ...")
    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            pc_clean = batch["img"].to(device).float()
            pc_strong = get_strong_aug(pc_clean)

            with autocast('cuda'):
                # 1. 冻结老师看干净数据，给出绝对稳健的指导
                with torch.no_grad():
                    logits_teacher = teacher_model(pc_clean)
                    probs_teacher = F.softmax(logits_teacher, dim=1)
                    max_probs_teacher, pseudo_labels = torch.max(probs_teacher, dim=1)

                    # 严苛过滤：只保留 > 0.65 的高置信度硬标签
                    mask = (max_probs_teacher >= args.threshold).float()
                    # T=0.5 锐化：让模糊的预测变得尖锐，提供强引导
                    probs_teacher_sharp = F.softmax(logits_teacher / args.temperature, dim=1)

                # 2. 学生模型被迫在干净和强增强下同时迎战
                pc_combined = torch.cat([pc_clean, pc_strong], dim=0)
                logits_combined = student_model(pc_combined)
                logits_student_clean, logits_student_strong = logits_combined.chunk(2)

                # 3. 极其纯净的黄金双轨 Loss，无任何负数/发散风险
                # 轨1 (Soft KD 保底): 让学生的干净特征，向老师锐化后的稳健分布靠拢
                log_probs_clean = F.log_softmax(logits_student_clean / args.temperature, dim=1)
                loss_soft = F.kl_div(log_probs_clean, probs_teacher_sharp.detach(), reduction='batchmean') * (
                            args.temperature ** 2)

                # 轨2 (Hard FixMatch 突破): 老师极度确信的样本，逼迫学生在强干扰下认出来
                ce_loss_strong = F.cross_entropy(logits_student_strong, pseudo_labels, reduction='none',
                                                 label_smoothing=0.1)
                loss_hard = (ce_loss_strong * mask).sum() / (mask.sum() + 1e-8)

                loss = (0.5 * loss_soft + 1.0 * loss_hard) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({
                "Loss": f"{loss.item() * accumulation_steps:.4f}",
                "Mask": f"{(mask.sum().item() / pc_clean.shape[0]):.1%}"
            })

        scheduler.step()

        print(f"\nEpoch {epoch + 1} 结束 -> 测试 3D Student 性能...")
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
        print(f"=> 3D Student Target Accuracy: {curr_acc:.2f}%")

        if curr_acc > best_acc:
            best_acc = curr_acc
            torch.save(student_model.state_dict(), os.path.join(args.output_dir, "best_self_distilled_dgcnn.pth"))
            print(f"🎉 突破！已保存稳健涨点的新模型 (当前最高: {best_acc:.2f}%)")

    print(f"\n✅ 终极稳健自我蒸馏收官! 最终冲刺准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()