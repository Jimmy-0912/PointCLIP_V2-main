import torch
import torch.nn as nn
from clip import clip


class GeometricFeatureExtractor(nn.Module):
    def __init__(self, num_bins=10):
        super().__init__()
        self.num_bins = num_bins

    @torch.no_grad()
    def forward(self, pc):
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


class CustomTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        # 依赖于精准的 argmax 寻找 EOS
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class DynamicPromptLearner(nn.Module):
    def __init__(self, clip_model, text_prompts, ctx_length=4, geo_dim=16):
        super().__init__()
        self.n_cls = len(text_prompts)
        self.ctx_length = ctx_length
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.dtype = clip_model.dtype
        device = next(clip_model.parameters()).device

        # 1. 提取基础 GPT 句子的静态 Embedding
        tokenized_prompts = clip.tokenize(text_prompts, truncate=True).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("class_embeddings", embedding)

        # 2. 动态 MLP 与安全阈值
        self.geo_extractor = GeometricFeatureExtractor(num_bins=10)
        self.meta_net = nn.Sequential(
            nn.Linear(geo_dim, self.ctx_dim // 2),
            nn.BatchNorm1d(self.ctx_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.ctx_dim // 2, ctx_length * self.ctx_dim)
        )

        # 极小初始化，外加一个可学习的 gamma 参数，使得起步非常平滑
        nn.init.normal_(self.meta_net[-1].weight, std=1e-5)
        nn.init.constant_(self.meta_net[-1].bias, 0)
        self.gamma = nn.Parameter(torch.tensor(0.01))

        self.text_encoder = CustomTextEncoder(clip_model)

    def forward(self, pc):
        B = pc.shape[0]

        geo_feats = self.geo_extractor(pc)
        # 引入 Tanh 把特征严格限制在 [-1, 1] 之间，防止无监督损失把它吹爆
        ctx = torch.tanh(self.meta_net(geo_feats))
        ctx = ctx.view(B, self.ctx_length, self.ctx_dim).type(self.dtype)

        prompts = self.class_embeddings.unsqueeze(0).expand(B, -1, -1, -1).clone()

        # 🚨 核心修复：纯纯的残差相加，完全不改变句子的 Token 位置 🚨
        prompts[:, :, 1:1 + self.ctx_length, :] = prompts[:, :, 1:1 + self.ctx_length, :] + self.gamma * ctx.unsqueeze(
            1)

        # 既然位置没变，原本的 ID 矩阵就可以直接用！(完美找准 EOS)
        base_tokens = self.tokenized_prompts.unsqueeze(0).expand(B, -1, -1)

        prompts = prompts.view(B * self.n_cls, 77, self.ctx_dim)
        base_tokens = base_tokens.reshape(B * self.n_cls, 77)

        text_features = self.text_encoder(prompts, base_tokens)
        text_features = text_features.view(B, self.n_cls, -1)

        return text_features

    @torch.no_grad()
    def forward_zs(self, B):
        """
        供知识蒸馏使用：直接输出未被 MLP 干扰的、最纯净的 GPT 特征
        """
        prompts = self.class_embeddings.unsqueeze(0).expand(B, -1, -1, -1)
        base_tokens = self.tokenized_prompts.unsqueeze(0).expand(B, -1, -1)

        prompts = prompts.reshape(B * self.n_cls, 77, self.ctx_dim)
        base_tokens = base_tokens.reshape(B * self.n_cls, 77)

        text_features = self.text_encoder(prompts, base_tokens)
        text_features = text_features.view(B, self.n_cls, -1)
        return text_features