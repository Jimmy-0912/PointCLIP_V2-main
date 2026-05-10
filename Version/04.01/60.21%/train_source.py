import os
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

from models.dgcnn import DGCNN


# ----------------- 简单的数据加载器 -----------------
class H5PointCloudDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        txt_file = os.path.join(root_dir, f'{split}_files.txt')

        self.all_data = []
        self.all_label = []

        with open(txt_file, "r") as f:
            for h5_name in f.readlines():
                # 兼容绝对路径和相对路径 (替换掉可能写死的 data/)
                h5_path = h5_name.strip()
                if h5_path.startswith('data/'):
                    h5_path = h5_path.replace('data/', 'datasets/', 1)

                # 如果还是找不到，尝试相对于 txt 文件的路径
                if not os.path.exists(h5_path):
                    h5_path = os.path.join(self.root_dir, os.path.basename(h5_path))

                f_h5 = h5py.File(h5_path, 'r')
                data = f_h5['data'][:].astype('float32')
                label = f_h5['label'][:].astype('int64')
                f_h5.close()
                self.all_data.append(data)
                self.all_label.append(label)

        self.all_data = np.concatenate(self.all_data, axis=0)
        self.all_label = np.concatenate(self.all_label, axis=0)
        print(f"Loaded {split} data: {self.all_data.shape[0]} samples")

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        points = self.all_data[idx]
        label = self.all_label[idx]
        # DGCNN 需要输入 [3, N] 格式，但模型内部我加了转置兼容，这里保持 [N, 3] 即可
        return torch.tensor(points), torch.tensor(label).squeeze()


# ----------------- 训练主循环 -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, required=True, help='ModelNet40 根目录')
    parser.add_argument('--output-dir', type=str, default='output/source_dgcnn')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载源域数据 (ModelNet40)
    train_dataset = H5PointCloudDataset(args.dataset_root, split='train')
    test_dataset = H5PointCloudDataset(args.dataset_root, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. 初始化 DGCNN (ModelNet40 是 40 类)
    model = DGCNN(num_classes=40).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 3. 监督训练
    best_acc = 0.0
    print("\n>>> 开始源域(ModelNet40)有监督预训练...")

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

        # 测试阶段
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
            save_path = os.path.join(args.output_dir, "dgcnn_source_modelnet40_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🎉 保存新的 Best Model: {save_path}")

    print(f"\n✅ 源域预训练完成！最高准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()