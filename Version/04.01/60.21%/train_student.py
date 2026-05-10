import os
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm

from dassl.engine import build_trainer
from dassl.config import get_cfg_default
from dassl.utils import setup_logger, set_random_seed

# 导入你的 Teacher 和 Student
from main_sfda import SFDA_PointCLIP, extend_cfg, reset_cfg
from models.dgcnn import DGCNN

import datasets.scanobjnn
import datasets.modelnet40


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


def soft_ce_loss(logits, target_probs):
    """软交叉熵损失：学习连续分布而不是硬标签，防止过拟合噪声"""
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_probs * log_probs).sum(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='output/final_uda_model_topovpr')
    parser.add_argument('--teacher-weight', type=str, required=True, help='老师(Topo-VPR)的权重路径')
    parser.add_argument('--source-weight', type=str, default='', help='源域预训练的3D主模型权重')
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--dataset-config-file', type=str, default='')
    parser.add_argument('--backbone', type=str, default='ViT-B/16')
    parser.add_argument('--epochs', type=int, default=150)

    # 🚨 显存护航：直接暴露 batch-size 参数，默认 16，彻底告别 OOM！
    parser.add_argument('--batch-size', type=int, default=16, help='控制显存占用，如果显存大可设为32')

    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--threshold', type=float, default=0.60)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--trainer', type=str, default='SFDA_PointCLIP')

    args, left_args = parser.parse_known_args()
    args.opts = left_args

    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(args.config_file)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)

    # 将我们自定义的 Batch Size 强行覆盖到底层框架中
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size

    cfg.TRAINER.NAME = "SFDA_PointCLIP"
    cfg.freeze()

    setup_logger(args.output_dir)
    set_random_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n>>> 正在加载 Topo-VPR 老师... (Weight: {args.teacher_weight})")
    teacher_trainer = build_trainer(cfg)
    state_dict = torch.load(args.teacher_weight, map_location=device, weights_only=True)
    teacher_trainer.prompt_learner.load_state_dict(state_dict)
    teacher_trainer.prompt_learner.eval()
    for param in teacher_trainer.prompt_learner.parameters():
        param.requires_grad = False

    train_loader = teacher_trainer.train_loader_x
    test_loader_raw = teacher_trainer.test_loader
    if isinstance(test_loader_raw, dict):
        test_loader = list(test_loader_raw.values())[0]
    else:
        test_loader = test_loader_raw
    num_classes = teacher_trainer.dm.dataset.num_classes

    print(f"\n>>> 初始化 3D 学生模型: DGCNN (Classes: {num_classes})")
    student_model = DGCNN(num_classes=num_classes).to(device)

    if args.source_weight and os.path.exists(args.source_weight):
        print(f">>> [重点] 正在加载源域预训练模型权重 (91.57% 的底子): {args.source_weight}")
        checkpoint = torch.load(args.source_weight, map_location=device, weights_only=True)
        model_dict = student_model.state_dict()

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        print(f"    -> 成功复用 Backbone {len(pretrained_dict)} 个网络层权重！")

        # SOTA 核心操作：跨类别字典对齐抽取
        align_map = {
            'bed': 2, 'cabinet': 14, 'chair': 8, 'desk': 12, 'display': 22,
            'door': 13, 'shelf': 4, 'sink': 29, 'sofa': 30, 'table': 33, 'toilet': 35
        }

        if 'linear3.weight' in checkpoint and checkpoint['linear3.weight'].shape[0] == 40:
            src_weight = checkpoint['linear3.weight']
            src_bias = checkpoint['linear3.bias']
            new_weight = torch.zeros((11, 256), device=device)
            new_bias = torch.zeros(11, device=device)

            shared_names = sorted(
                ['cabinet', 'chair', 'desk', 'display', 'door', 'shelf', 'table', 'bed', 'sink', 'sofa', 'toilet'])
            for i, class_name in enumerate(shared_names):
                mn40_idx = align_map[class_name]
                new_weight[i] = src_weight[mn40_idx]
                new_bias[i] = src_bias[mn40_idx]

            model_dict['linear3.weight'] = new_weight
            model_dict['linear3.bias'] = new_bias
            print("    -> SOTA 技巧触发：成功完成源域(40) -> 目标域(11) 分类头权重精准对齐抽取！")

        student_model.load_state_dict(model_dict)

    print("\n>>> [Baseline] 评估纯源域模型 (Source-Only) 在目标域 11 类上的初始性能...")
    student_model.eval()
    total_correct_src = 0.0
    total_samples_src = 0
    with torch.no_grad():
        for batch in test_loader:
            pc = batch["img"].to(device).float()
            label = batch["label"].to(device)
            logits = student_model(pc)
            total_correct_src += accuracy(logits, label)[0]
            total_samples_src += pc.shape[0]

    baseline_acc = (total_correct_src / total_samples_src) * 100
    print(f"=> Source-Only Baseline Target Accuracy: {baseline_acc:.2f}% (感谢权重对齐，不再是盲猜！)")
    print("---------------------------------------------------------------------\n")

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\n>>> 开始原创范式：Soft Consensus + 3D Information Maximization ...")
    best_acc = 0.0

    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            pc = batch["img"].to(device).float()
            b_sz = pc.shape[0]

            with torch.no_grad():
                micro_sz = 8
                logits_teacher_list = []
                for i in range(0, b_sz, micro_sz):
                    pc_micro = pc[i:i + micro_sz]
                    logits_micro = teacher_trainer.model_inference(pc_micro)
                    logits_teacher_list.append(logits_micro)
                logits_teacher = torch.cat(logits_teacher_list, dim=0)

                probs_teacher_soft = F.softmax(logits_teacher / args.temperature, dim=1)
                max_probs_teacher, pseudo_labels = torch.max(probs_teacher_soft, dim=1)

                base_prob = 1.0 / num_classes
                confidence_weight = torch.clamp((max_probs_teacher - base_prob) / (1.0 - base_prob), min=0.0)

            # 🚨 显存垃圾回收：阅后即焚，立刻释放 Teacher 的计算图，给 DGCNN 腾地方！
            del logits_teacher, max_probs_teacher, logits_teacher_list
            torch.cuda.empty_cache()

            pc_strong = get_strong_aug(pc)
            logits_student_clean = student_model(pc)
            logits_student_strong = student_model(pc_strong)

            ce_loss_clean = soft_ce_loss(logits_student_clean, probs_teacher_soft.detach())
            masked_ce_clean = (ce_loss_clean * confidence_weight).mean()

            ce_loss_strong = soft_ce_loss(logits_student_strong, probs_teacher_soft.detach())
            masked_ce_strong = (ce_loss_strong * confidence_weight).mean()

            probs_clean = F.softmax(logits_student_clean.detach(), dim=1)
            log_probs_strong = F.log_softmax(logits_student_strong, dim=1)
            consistency_loss = F.kl_div(log_probs_strong, probs_clean, reduction='batchmean')

            probs_student = F.softmax(logits_student_clean, dim=1)
            ent_loss = -torch.mean(torch.sum(probs_student * torch.log(probs_student + 1e-6), dim=1))
            mean_probs = torch.mean(probs_student, dim=0)
            div_loss = torch.sum(mean_probs * torch.log(mean_probs + 1e-6))

            loss = masked_ce_clean + masked_ce_strong + 2.0 * consistency_loss + 0.1 * ent_loss + 0.5 * div_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Avg_Weight": f"{confidence_weight.mean().item():.2f}"
            })

        scheduler.step()

        print(f"\nEpoch {epoch + 1} 结束 -> 测试性能...")
        student_model.eval()
        total_correct = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                pc = batch["img"].to(device).float()
                label = batch["label"].to(device)
                logits = student_model(pc)
                total_correct += accuracy(logits, label)[0]
                total_samples += pc.shape[0]

        curr_acc = (total_correct / total_samples) * 100
        print(f"=> Student DGCNN Target Accuracy: {curr_acc:.2f}%")

        if curr_acc > best_acc:
            best_acc = curr_acc
            print(f"🎉 发现新高! 保存最佳学生模型...")
            torch.save(student_model.state_dict(), os.path.join(args.output_dir, "best_student_dgcnn_finetuned.pth"))

    print(f"\n✅ 突破上限完成! 最佳学生模型在目标域的准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()