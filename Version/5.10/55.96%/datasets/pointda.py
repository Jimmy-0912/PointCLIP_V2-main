import os
import glob
import numpy as np
import open3d as o3d
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class PointDA(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = cfg.DATASET.ROOT
        # PointDA-10 共享的 10 个标准类别
        self.new_classnames = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa',
                               'table']

        print(f"\n>>> 正在加载 PointDA-10 数据集 (PLY格式)... 路径: {self.dataset_dir}")
        train = self.read_data('train')
        test = self.read_data('test')

        super().__init__(train_x=train, val=test, test=test)
        self._classnames = self.new_classnames
        self._num_classes = len(self.new_classnames)

    def read_data(self, split):
        items = []
        for label, classname in enumerate(self.new_classnames):
            split_dir = os.path.join(self.dataset_dir, classname, split)
            if not os.path.exists(split_dir):
                continue

            ply_files = glob.glob(os.path.join(split_dir, "*.ply"))
            for ply_path in ply_files:
                # 1. 使用 Open3D 读取 PLY 文件
                try:
                    pcd = o3d.io.read_point_cloud(ply_path)
                    points = np.asarray(pcd.points).astype('float32')
                except Exception as e:
                    print(f"读取文件失败: {ply_path}, 错误: {e}")
                    continue

                num_points = points.shape[0]
                if num_points == 0:
                    continue

                # 2. 统一采样到 1024 个点
                if num_points >= 1024:
                    choice = np.random.choice(num_points, 1024, replace=False)
                else:
                    choice = np.random.choice(num_points, 1024, replace=True)
                points = points[choice, :]

                # 3. 极其重要的归一化 (Zero-mean, Unit-sphere)
                centroid = np.mean(points, axis=0)
                points = points - centroid
                m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
                points = points / (m + 1e-8)

                # 4. 封装进 Dassl 的 Datum 格式 (直接传入 array 数据)
                item = Datum(
                    impath=points,
                    label=label,
                    classname=classname
                )
                items.append(item)
        return items