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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='output/student_dgcnn')
    parser.add_argument('--teacher-weight', type=str, required=True, help='老师的权重路径')
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--dataset-config-file', type=str, default='')
    parser.add_argument('--backbone', type=str, default='ViT-B/16')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=2.0, help='知识蒸馏温度')

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--trainer', type=str, default='SFDA_PointCLIP')

    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(args.config_file)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)

    cfg.TRAINER.NAME = "SFDA_PointCLIP"
    cfg.freeze()

    setup_logger(args.output_dir)
    set_random_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化教师模型
    print(f"\n>>> 正在加载知识蒸馏教师模型... (Weight: {args.teacher_weight})")
    teacher_trainer = build_trainer(cfg)

    # 加入 weights_only=True 消除 PyTorch 安全警告
    state_dict = torch.load(args.teacher_weight, map_location=device, weights_only=True)
    teacher_trainer.prompt_learner.meta_net.load_state_dict(state_dict)

    # 彻底冻结教师
    teacher_trainer.prompt_learner.eval()
    for param in teacher_trainer.prompt_learner.parameters():
        param.requires_grad = False

    # 2. 准备数据
    train_loader = teacher_trainer.train_loader_x

    # 🚨 核心兼容性修复：智能判断 test_loader 是字典还是实体 DataLoader 🚨
    test_loader_raw = teacher_trainer.test_loader
    if isinstance(test_loader_raw, dict):
        test_loader = list(test_loader_raw.values())[0]
    else:
        test_loader = test_loader_raw

    num_classes = teacher_trainer.dm.dataset.num_classes

    # 3. 初始化学生模型 DGCNN
    print(f"\n>>> 初始化 3D 纯净学生模型: DGCNN (Classes: {num_classes})")
    student_model = DGCNN(num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. 开始软标签蒸馏
    print("\n>>> 开始跨模态伪标签知识蒸馏 (Inference Micro-batching Protected)...")
    best_acc = 0.0
    T = args.temperature

    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            pc = batch["img"].to(device).float()

            # --- 教师前向传播产生高质量的软目标 (Soft Targets) ---
            with torch.no_grad():
                b_sz = pc.shape[0]
                micro_sz = 4  # 调到极小，绝对防止在推理教师模型时 OOM
                logits_teacher_list = []

                for i in range(0, b_sz, micro_sz):
                    pc_micro = pc[i:i + micro_sz]
                    logits_micro = teacher_trainer.model_inference(pc_micro)
                    logits_teacher_list.append(logits_micro)

                logits_teacher = torch.cat(logits_teacher_list, dim=0)

            # --- 学生原生 3D 前向传播 ---
            logits_student = student_model(pc)

            # 知识蒸馏 KL Loss
            soft_teacher = F.softmax(logits_teacher / T, dim=1)
            log_soft_student = F.log_softmax(logits_student / T, dim=1)

            loss = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean') * (T ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"KD_Loss": f"{loss.item():.4f}"})

        scheduler.step()

        # 5. 测试阶段：仅用真实的 Label 检验学生的能力
        print(f"\nEpoch {epoch + 1} 结束, 测试学生模型在目标域的真实性能...")
        student_model.eval()
        total_correct = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                pc = batch["img"].to(device).float()
                label = batch["label"].to(device)

                logits = student_model(pc)
                acc1 = accuracy(logits, label)[0]

                total_correct += acc1
                total_samples += pc.shape[0]

        curr_acc = (total_correct / total_samples) * 100
        print(f"=> Student DGCNN Accuracy: {curr_acc:.2f}% (Teacher was ~66%)")

        if curr_acc > best_acc:
            best_acc = curr_acc
            print(f"🎉 发现新高! 保存最佳学生模型...")
            torch.save(student_model.state_dict(), os.path.join(args.output_dir, "best_student_dgcnn.pth"))

    print(f"\n✅ 蒸馏完成! 最佳学生模型准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()