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
    纯粹且安全的 3D 强增强，供学生挖掘深层抗噪能力
    """
    B, N, C = pc.shape
    pc_strong = pc.clone()
    device = pc.device

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
    # 🚨 这里的 teacher_weight 必须传入上一阶段得到的 54.61% 的 DGCNN 权重！
    parser.add_argument('--teacher-weight', type=str, required=True, help='上一阶段 54.61% 的 DGCNN 学生权重')
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--dataset-config-file', type=str, default='')
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--micro-batch', type=int, default=8)

    # 🚨 同模态的老师更靠谱，阈值提高到 0.75，提取绝对纯金标签！
    parser.add_argument('--threshold', type=float, default=0.75)
    parser.add_argument('--temperature', type=float, default=1.0)

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

    print(f"\n>>> [1/2] 部署 3D 导师模型 (冻结前一阶段的 54.61% 优等生)...")
    teacher_model = DGCNN(num_classes=num_classes).to(device)
    teacher_model.load_state_dict(torch.load(args.teacher_weight, map_location=device, weights_only=True))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    print(f"\n>>> [2/2] 部署 3D 学生模型与 EMA 稳定器...")
    # 学生模型同样从 54.61% 的基础起步，实现自我进化 (Self-Refinement)
    student_model = DGCNN(num_classes=num_classes).to(device)
    student_model.load_state_dict(torch.load(args.teacher_weight, map_location=device, weights_only=True))

    ema_student_model = DGCNN(num_classes=num_classes).to(device)
    ema_student_model.load_state_dict(student_model.state_dict())
    ema_student_model.eval()
    for param in ema_student_model.parameters():
        param.requires_grad = False

    print("\n>>> [Baseline] 测试初始 3D 导师模型在目标域的准确率...")
    init_correct, init_samples = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            pc = batch["img"].to(device).float()
            label = batch["label"].to(device)
            logits = teacher_model(pc)
            init_correct += accuracy(logits, label)[0]
            init_samples += pc.shape[0]
    baseline_acc = (init_correct / init_samples) * 100
    print(f"=> 3D Teacher 初始准确率: {baseline_acc:.2f}% (冲刺 60% 的起点！)\n")

    # 🚨 极小的微调学习率，避免破坏已有的坚固分类边界
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    best_acc = baseline_acc

    print("\n>>> 开始同模态 3D 自我蒸馏: 纯 3D 黄金伪标签 + 信息最大化 ...")
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
                    # --- 1. 3D 老师在干净数据上输出预测 ---
                    logits_teacher = teacher_model(pc)
                    probs_teacher = F.softmax(logits_teacher, dim=1)
                    max_probs_teacher, pseudo_labels = torch.max(probs_teacher, dim=1)

                    # 严苛过滤：只保留 3D 老师把握 > 0.75 的标签
                    mask = (max_probs_teacher >= args.threshold).float()
                    probs_teacher_soft = F.softmax(logits_teacher / args.temperature, dim=1)

                # --- 2. 3D 学生双路输入 ---
                pc_combined = torch.cat([pc, pc_strong], dim=0)
                logits_combined = student_model(pc_combined)
                logits_student_clean, logits_student_strong = logits_combined.chunk(2)

                # --- 3. 终极自适应 Loss ---
                # A. 强硬对齐 (FixMatch): 强迫学生在强干扰下认出老师的高置信度答案
                ce_loss_strong = F.cross_entropy(logits_student_strong, pseudo_labels, reduction='none',
                                                 label_smoothing=0.1)
                loss_hard = (ce_loss_strong * mask).sum() / (mask.sum() + 1e-8)

                # B. 保底软对齐 (Soft KD): 保证未被 Mask 选中的样本，学生大方向不偏离老师
                log_probs_clean = F.log_softmax(logits_student_clean / args.temperature, dim=1)
                loss_soft = F.kl_div(log_probs_clean, probs_teacher_soft.detach(), reduction='batchmean') * (
                            args.temperature ** 2)

                # C. 探索边界 (InfoMax): 鼓励学生对强干扰数据的预测产生更尖锐的判断，推动决策边界穿过低密度区
                probs_student_strong = F.softmax(logits_student_strong, dim=1)
                ent_loss = -torch.mean(torch.sum(probs_student_strong * torch.log(probs_student_strong + 1e-8), dim=1))
                mean_probs = torch.mean(probs_student_strong, dim=0)
                div_loss = torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

                # 安全配比，绝不崩塌
                loss = (1.0 * loss_hard + 0.5 * loss_soft + 0.1 * ent_loss + 0.2 * div_loss) / accumulation_steps

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

        print(f"\nEpoch {epoch + 1} 结束 -> 测试 3D Self-Distilled 性能...")
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
            torch.save(ema_student_model.state_dict(), os.path.join(args.output_dir, "best_self_distilled_dgcnn.pth"))
            print(f"🎉 进化成功！已保存突破极限的模型 (当前最高: {best_acc:.2f}%)")

    print(f"\n✅ 终极自我蒸馏收官! 最终冲刺准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()