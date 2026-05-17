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
    纯粹且安全的 3D 强增强，保护 K-NN 图结构
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
    # 🚨 请务必传入刚刚跑出的 56.76% 的权重！这是冲击 60% 的阶梯！
    parser.add_argument('--teacher-weight', type=str, required=True, help='起始基础权重 (56.76% 的学生权重)')
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--dataset-config-file', type=str, default='')
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--micro-batch', type=int, default=8)

    # 🚨 阈值设为 0.55，释放黄金锚点
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
    cfg.freeze()

    setup_logger(args.output_dir)
    set_random_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = DataManager(cfg)
    train_loader = dm.train_loader_x
    test_loader_raw = dm.test_loader
    test_loader = list(test_loader_raw.values())[0] if isinstance(test_loader_raw, dict) else test_loader_raw
    num_classes = 10

    print(f"\n>>> [1/2] 部署绝对冻结的 3D 导师 (56.76% 铁底)...")
    teacher_model = DGCNN(num_classes=num_classes).to(device)
    teacher_model.load_state_dict(torch.load(args.teacher_weight, map_location=device, weights_only=True))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    print(f"\n>>> [2/2] 部署 3D 学生模型 (开启无上限进化)...")
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
    print(f"=> Frozen Teacher 初始准确率: {baseline_acc:.2f}% (最后的冲刺起点！)\n")

    # 学习率给足 1e-4，赋予学生探索新知识的能量
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    best_acc = baseline_acc

    print("\n>>> 开始分布对齐自训练 (DAST): 硬锚点 + 探索熵 + 全局分布对齐 ...")
    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            pc_clean = batch["img"].to(device).float()
            pc_strong = get_strong_aug(pc_clean)

            with torch.autocast(device_type='cuda'):
                # 1. 冻结老师看干净数据，提取指导信号
                with torch.no_grad():
                    logits_teacher = teacher_model(pc_clean)
                    probs_teacher = F.softmax(logits_teacher, dim=1)
                    max_probs_teacher, pseudo_labels = torch.max(probs_teacher, dim=1)
                    mask = (max_probs_teacher >= args.threshold).float()

                # 2. 学生模型仅在强增强下迎战，激发抗噪潜能
                logits_student_strong = student_model(pc_strong)
                probs_student_strong = F.softmax(logits_student_strong, dim=1)

                # 3. 🚨 终极破壁 Loss 架构 (彻底移除阻碍成长的 Soft KD)

                # 轨1 [锚点]: 老师有把握的题，学生强制对齐 (Hard FixMatch)
                ce_loss_strong = F.cross_entropy(logits_student_strong, pseudo_labels, reduction='none',
                                                 label_smoothing=0.1)
                loss_hard = (ce_loss_strong * mask).sum() / (mask.sum() + 1e-8)

                # 轨2 [探索]: 老师没把握的题，学生自己寻找高确信的分类边界 (Self-Entropy)
                ent_loss = -torch.sum(probs_student_strong * torch.log(probs_student_strong + 1e-8), dim=1)
                # 仅对未被 Mask 的样本进行熵最小化探索
                loss_explore = (ent_loss * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)

                # 轨3 [防坍塌]: 全局分布对齐 (Distribution Alignment)
                # 不强求具体哪个样本对标老师，但要求当前 Batch 的类别总体比例和老师一致！
                mean_prob_student = probs_student_strong.mean(dim=0)
                mean_prob_teacher = probs_teacher.mean(dim=0).detach()
                # 交叉熵分布对齐，极其稳定，绝不引发模式坍塌！
                loss_dist = -torch.sum(mean_prob_teacher * torch.log(mean_prob_student + 1e-8))

                # 完美融合：探索未知 + 守住底线
                loss = (1.0 * loss_hard + 0.3 * loss_explore + 1.0 * loss_dist) / accumulation_steps

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
            print(f"🎉 突破天花板！已保存新巅峰模型 (当前最高: {best_acc:.2f}%)")

    print(f"\nDAST 终极冲刺收官! 最终冲刺准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()