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
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class DynamicPromptLearner(nn.Module):
    def __init__(self, clip_model, text_prompts, ctx_length=4, geo_dim=16, num_views=10):
        super().__init__()
        self.n_cls = len(text_prompts)
        self.ctx_length = ctx_length
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.dtype = clip_model.dtype
        device = next(clip_model.parameters()).device

        tokenized_prompts = clip.tokenize(text_prompts, truncate=True).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("class_embeddings", embedding)

        self.geo_extractor = GeometricFeatureExtractor(num_bins=10)

        # 动态 Context 生成网络
        self.meta_net = nn.Sequential(
            nn.Linear(geo_dim, self.ctx_dim // 2),
            nn.BatchNorm1d(self.ctx_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.ctx_dim // 2, ctx_length * self.ctx_dim)
        )

        # 3D拓扑引导的动态视图路由网络 (Topological View Router)
        self.view_router = nn.Sequential(
            nn.Linear(geo_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_views)
        )

        nn.init.normal_(self.meta_net[-1].weight, std=1e-5)
        nn.init.constant_(self.meta_net[-1].bias, 0)
        # 初始化视图权重为接近均匀分布
        nn.init.normal_(self.view_router[-1].weight, std=1e-5)
        nn.init.constant_(self.view_router[-1].bias, 0)

        self.gamma = nn.Parameter(torch.tensor(0.01))
        self.text_encoder = CustomTextEncoder(clip_model)

    def forward(self, pc):
        B = pc.shape[0]

        # 提取 3D 拓扑特征
        geo_feats = self.geo_extractor(pc)

        # TVR: 预测动态视图权重 [B, 10]
        dynamic_view_weights = torch.softmax(self.view_router(geo_feats), dim=1)

        # TSOP: 生成正交的动态提示词
        ctx = self.meta_net(geo_feats)
        ctx = ctx.view(B, self.ctx_length, self.ctx_dim).type(self.dtype)

        semantic_anchor = self.class_embeddings[0, 0, :].view(1, 1, self.ctx_dim)

        # 将 ctx 投影到 semantic_anchor 的零空间中
        projection = (ctx * semantic_anchor).sum(dim=-1, keepdim=True) / (
                    semantic_anchor.norm() ** 2 + 1e-8) * semantic_anchor
        ctx_ortho = ctx - projection

        prompts = self.class_embeddings.unsqueeze(0).expand(B, -1, -1, -1).clone()
        prompts[:, :, 1:1 + self.ctx_length, :] = prompts[
                                                      :, :, 1:1 + self.ctx_length, :] + self.gamma * ctx_ortho.unsqueeze(
            1)

        base_tokens = self.tokenized_prompts.unsqueeze(0).expand(B, -1, -1)

        prompts = prompts.view(B * self.n_cls, 77, self.ctx_dim)
        base_tokens = base_tokens.reshape(B * self.n_cls, 77)

        text_features = self.text_encoder(prompts, base_tokens)
        text_features = text_features.view(B, self.n_cls, -1)

        return text_features, dynamic_view_weights

    @torch.no_grad()
    def forward_zs(self, B):
        prompts = self.class_embeddings.unsqueeze(0).expand(B, -1, -1, -1)
        base_tokens = self.tokenized_prompts.unsqueeze(0).expand(B, -1, -1)

        prompts = prompts.reshape(B * self.n_cls, 77, self.ctx_dim)
        base_tokens = base_tokens.reshape(B * self.n_cls, 77)

        text_features = self.text_encoder(prompts, base_tokens)
        text_features = text_features.view(B, self.n_cls, -1)
        return text_features