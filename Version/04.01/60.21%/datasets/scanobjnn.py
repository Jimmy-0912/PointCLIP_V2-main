import os
import h5py
import numpy as np
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class ScanObjectNN(DatasetBase):

    def __init__(self, cfg):

        self.dataset_dir = cfg.DATASET.ROOT

        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        classnames = self.read_classnames(text_file)

        # 🚨 论文核心标准对齐：SOTA Domain Adaptation 闭集设定 🚨
        # 借鉴 SOTA：必须对共享类别进行【字母顺序排序 (Alphabetical Sorting)】
        # 这样才能保证在源域和目标域之间建立绝对一致的 0~10 索引映射，防止分类头权重抽取时发生错位！
        shared_names = sorted([
            'cabinet', 'chair', 'desk', 'display', 'door',
            'shelf', 'table', 'bed', 'sink', 'sofa', 'toilet'
        ])

        # 建立新旧标签的映射表 (严格按照字母顺序分配 0~10)
        self.valid_mapping = {}
        self.new_classnames = shared_names

        for old_idx, name in classnames.items():
            if name in shared_names:
                new_idx = shared_names.index(name)
                self.valid_mapping[old_idx] = new_idx

        train_data, train_label = self.load_data(os.path.join(self.dataset_dir, 'train_files.txt'))
        test_data, test_label = self.load_data(os.path.join(self.dataset_dir, 'test_files.txt'))

        # 读取数据时，自动过滤掉非重叠类 (如 bag, box, bin, pillow)
        train = self.read_data(classnames, train_data, train_label)
        test = self.read_data(classnames, test_data, test_label)

        super().__init__(train_x=train, val=test, test=test)

        # 🚨 修复 Dassl 底层只读属性报错：改用内部变量 _classnames 和 _num_classes 🚨
        # 覆盖底层的类别名称，让 CLIP 知道只生成这 11 个类别的 Prompt
        self._classnames = self.new_classnames
        self._num_classes = len(self.new_classnames)

    def load_data(self, data_path):
        all_data = []
        all_label = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                all_data.append(data)
                all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                classname = line.strip()
                classnames[i] = classname
        return classnames

    def read_data(self, classnames, datas, labels):
        items = []

        for i, data in enumerate(datas):
            label = int(labels[i])

            # 如果是重叠类，则加入训练/评估；否则直接丢弃
            if label in self.valid_mapping:
                classname = classnames[label]
                new_label = self.valid_mapping[label]  # 转换为 0~10 的新索引

                item = Datum(
                    impath=data,
                    label=new_label,
                    classname=classname
                )
                items.append(item)

        return items