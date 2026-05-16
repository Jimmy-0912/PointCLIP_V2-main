import torch
import torch.nn as nn


class GeometricFeatureExtractor(nn.Module):
    def __init__(self, num_bins=10):
        super().__init__()
        self.num_bins = num_bins

    @torch.no_grad()
    def forward(self, pc):
        # 适配不同维度输入
        if pc.dim() == 3 and pc.shape[1] == 3 and pc.shape[2] > 3:
            pc = pc.transpose(1, 2)

        B, N, _ = pc.shape

        pc_max = torch.max(pc, dim=1)[0]
        pc_min = torch.min(pc, dim=1)[0]
        pc_range = pc_max - pc_min

        pc_std = torch.std(pc, dim=1, unbiased=False) + 1e-6
        pc_mean = torch.mean(pc, dim=1)

        dist = torch.norm(pc - pc_mean.unsqueeze(1), dim=-1)
        max_dist = torch.max(dist, dim=1, keepdim=True)[0] + 1e-6
        norm_dist = dist / max_dist

        histograms = []
        for i in range(B):
            hist = torch.histc(norm_dist[i], bins=self.num_bins, min=0.0, max=1.0)
            hist = hist / N
            histograms.append(hist)
        dist_hist = torch.stack(histograms)

        geo_features = torch.cat([pc_range, pc_std, dist_hist], dim=1)
        return geo_features


class TopologicalViewRouter(nn.Module):
    """
    核心创新：放弃修改文本，专注于 3D 引导的 2D 视图路由！
    """

    def __init__(self, geo_dim=16, num_views=10):
        super().__init__()
        self.geo_extractor = GeometricFeatureExtractor(num_bins=10)

        # 极简且高效的路由网络
        self.router = nn.Sequential(
            nn.Linear(geo_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_views)
        )

        # 🚨 关键初始化：让起始的视图权重绝对平均，等价于 Zero-Shot 基线！
        nn.init.normal_(self.router[-1].weight, std=0.001)
        nn.init.constant_(self.router[-1].bias, 0)

    def forward(self, pc):
        # 1. 提取 3D 特征
        geo_feats = self.geo_extractor(pc)

        # 2. 输出动态权重 (Softmax 保证和为 1)
        # 输出维度: [B, 10]
        view_weights = torch.softmax(self.router(geo_feats), dim=1)

        return view_weights