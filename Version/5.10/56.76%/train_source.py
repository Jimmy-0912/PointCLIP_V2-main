import os
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import open3d as o3d

from models.dgcnn import DGCNN


class PointDAPlyDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.classnames = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa',
                           'table']

        self.filepaths = []
        self.labels = []

        for label, classname in enumerate(self.classnames):
            split_dir = os.path.join(root_dir, classname, split)
            if not os.path.exists(split_dir):
                continue
            ply_files = glob.glob(os.path.join(split_dir, "*.ply"))
            for f in ply_files:
                self.filepaths.append(f)
                self.labels.append(label)

        print(f"Loaded {split} data: {len(self.filepaths)} samples")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        ply_path = self.filepaths[idx]
        label = self.labels[idx]

        # 读取 PLY 文件
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points).astype('float32')

        # 统一采样到 1024 个点
        num_points = points.shape[0]
        if num_points >= 1024:
            choice = np.random.choice(num_points, 1024, replace=False)
        else:
            choice = np.random.choice(num_points, 1024, replace=True)
        points = points[choice, :]

        # 极其重要的归一化 (Zero-mean, Unit-sphere)
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / (m + 1e-8)

        return torch.tensor(points), torch.tensor(label).long()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, required=True, help='PointDA源域根目录(如 modelnet)')
    parser.add_argument('--output-dir', type=str, default='output/source_dgcnn_pointda10')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PointDAPlyDataset(args.dataset_root, split='train')
    test_dataset = PointDAPlyDataset(args.dataset_root, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 🚨 更改为 10 分类！
    model = DGCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    print("\n>>> 开始源域(PointDA-10)有监督预训练...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for points, labels in pbar:
            points, labels = points.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for points, labels in test_loader:
                points, labels = points.to(device), labels.to(device)
                logits = model(points)
                preds = logits.max(1)[1]
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        acc = 100. * correct / total
        print(f"Epoch {epoch + 1} Test Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(args.output_dir, "dgcnn_source_best.pth")
            torch.save(model.state_dict(), save_path)

    print(f"\n✅ 源域预训练完成！最高准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()