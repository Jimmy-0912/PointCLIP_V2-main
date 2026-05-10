import os
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm

from dassl.engine import build_trainer, TRAINER_REGISTRY, TrainerX
from dassl.config import get_cfg_default
from dassl.utils import setup_logger, set_random_seed

from clip import clip
from trainers.mv_utils_zs import Realistic_Projection
# 🚨 必须导入 best_prompt_weight，以获取 GPT-3 预生成的提示词字典
from trainers.best_param import best_prompt_weight
from dynamic_prompt import DynamicPromptLearner

import datasets.scanobjnn
import datasets.modelnet40


@TRAINER_REGISTRY.register()
class SFDA_PointCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"

        clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME, device=device)
        clip_model.float()
        clip_model.eval()

        for param in clip_model.parameters():
            param.requires_grad = False

        self.pc_views = Realistic_Projection()

        dataset_name = cfg.DATASET.NAME.lower()
        raw_name = cfg.MODEL.BACKBONE.NAME.lower().replace('-', '_').replace('/', '')
        bb_name = getattr(cfg.MODEL.BACKBONE, 'NAME2', raw_name)
        if bb_name == "rn50" and "vit" in raw_name:
            bb_name = raw_name

        prompt_key = f"{dataset_name}_{bb_name}_test_prompts"

        if prompt_key in best_prompt_weight:
            # 🚨 挽救点云 SOTA 性能的关键修复：精准对齐 GPT-3 提示词 🚨
            original_prompts = best_prompt_weight[prompt_key]

            # 这是 ScanObjectNN 原始的 15 个类别顺序
            orig_names = ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 'door', 'shelf', 'table', 'bed',
                          'pillow', 'sink', 'sofa', 'toilet']
            classnames = self.dm.dataset.classnames  # 这里是我们新排好序的 11 类

            text_prompts = []
            for name in classnames:
                orig_idx = orig_names.index(name)
                text_prompts.append(original_prompts[orig_idx])
            print(f"\n>>> [对齐成功] 已成功抽取并重排 11 类的 GPT-3 动态提示词！")
        else:
            classnames = self.dm.dataset.classnames
            # 兜底：如果找不到权重，必须使用 depth map
            text_prompts = [f"a depth map of a {name}." for name in classnames]

        self.num_views = getattr(cfg.MODEL.PROJECT, 'NUM_VIEWS', 10)

        self.prompt_learner = DynamicPromptLearner(
            clip_model, text_prompts, ctx_length=4, geo_dim=16, num_views=self.num_views
        ).to(device)

        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.model = self.prompt_learner

        self.optim = torch.optim.AdamW(
            self.prompt_learner.parameters(),
            lr=1e-4,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=cfg.OPTIM.MAX_EPOCH)

    def train(self):
        print(f"\n>>> [Sanity Check] {self.cfg.MODEL.BACKBONE.NAME} 初始 11 类 Zero-Shot 性能摸底...")
        self.test()

        print("\n>>> 开始论文终极方案：基于 MCF 边界过滤的 Topo-VPR...")
        target_loader = self.train_loader_x

        for epoch in range(self.cfg.OPTIM.MAX_EPOCH):
            self.prompt_learner.train()
            total_loss = 0.0
            total_mask_ratio = 0.0

            pbar = tqdm(target_loader, desc=f"Epoch {epoch + 1}/{self.cfg.OPTIM.MAX_EPOCH}")

            for batch_idx, batch in enumerate(pbar):
                loss_summary = self.forward_backward(batch)
                total_loss += loss_summary["loss"]
                total_mask_ratio += loss_summary.get("mask_ratio", 0)
                pbar.set_postfix(loss_summary)

            self.sched.step()
            avg_mask = total_mask_ratio / len(target_loader)
            print(
                f"Epoch {epoch + 1} 结束. Avg Loss: {total_loss / len(target_loader):.4f}, 绝对纯净锚点比例: {avg_mask:.1%}")

            self.test()

            save_path = os.path.join(self.cfg.OUTPUT_DIR, f"sfda_mlp_epoch_{epoch + 1}.pth")
            torch.save(self.prompt_learner.state_dict(), save_path)

    def forward_backward(self, batch):
        pc = batch["img"].cuda().float()
        if pc.dim() == 4:
            pc = pc.squeeze(1)

        B = pc.shape[0]
        micro_batch_size = 4
        device = pc.device

        self.optim.zero_grad()
        total_loss_val = 0.0
        total_mask_val = 0.0

        for i in range(0, B, micro_batch_size):
            pc_micro = pc[i:i + micro_batch_size]
            b_micro = pc_micro.shape[0]

            with torch.no_grad():
                images = self.pc_views.get_img(pc_micro).cuda()
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=True)
                images = images.type(self.dtype)

                image_feat = self.clip_model.visual(images)
                image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
                channel = image_feat.shape[-1]

                text_feat_zs = self.prompt_learner.forward_zs(b_micro)
                text_feat_zs = text_feat_zs / (text_feat_zs.norm(dim=-1, keepdim=True) + 1e-6)
                text_feat_zs = text_feat_zs.repeat(1, 1, self.num_views)

                uniform_weights = torch.ones(b_micro, self.num_views, 1, device=device)
                image_feat_zs_w = image_feat.reshape(-1, self.num_views, channel) * uniform_weights
                image_feat_zs_w = image_feat_zs_w.reshape(-1, self.num_views * channel).type(self.dtype)

                cos_sim = torch.bmm(image_feat_zs_w.unsqueeze(1), text_feat_zs.transpose(1, 2)).squeeze(1)

                top2_sim, top2_idx = torch.topk(cos_sim, 2, dim=1)
                margin = top2_sim[:, 0] - top2_sim[:, 1]
                pseudo_labels = top2_idx[:, 0]

                mask = (margin > 0.03).float()

            text_feat, dynamic_view_weights = self.prompt_learner(pc_micro)
            dynamic_view_weights = dynamic_view_weights * self.num_views

            image_feat_w = image_feat.reshape(-1, self.num_views, channel) * dynamic_view_weights.reshape(b_micro,
                                                                                                          self.num_views,
                                                                                                          1)
            image_feat_w = image_feat_w.reshape(-1, self.num_views * channel).type(self.dtype)

            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
            text_feat = text_feat.repeat(1, 1, self.num_views)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * torch.bmm(image_feat_w.unsqueeze(1), text_feat.transpose(1, 2)).squeeze(1)
            probs = F.softmax(logits, dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
            masked_ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            masked_ent_loss = (entropy * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)

            mean_probs = torch.mean(probs, dim=0)
            div_loss = torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            if mask.sum() > 0:
                loss = 1.0 * masked_ce_loss + 0.1 * masked_ent_loss + 0.5 * div_loss
            else:
                loss = 0.5 * masked_ent_loss + 0.5 * div_loss

            loss.backward()
            total_loss_val += loss.item()
            total_mask_val += (mask.sum().item() / b_micro) * (b_micro / B)

        torch.nn.utils.clip_grad_norm_(self.prompt_learner.parameters(), max_norm=1.0)
        self.optim.step()

        return {
            "loss": total_loss_val,
            "mask_ratio": total_mask_val
        }

    def model_inference(self, pc, label=None):
        if pc.dim() == 4:
            pc = pc.squeeze(1)

        text_feat, dynamic_view_weights = self.prompt_learner(pc)
        dynamic_view_weights = dynamic_view_weights * self.num_views

        images = self.pc_views.get_img(pc).cuda()
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=True)
        images = images.type(self.dtype)

        image_feat = self.clip_model.visual(images)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        channel = image_feat.shape[-1]
        b_micro = pc.shape[0]
        image_feat_w = image_feat.reshape(-1, self.num_views, channel) * dynamic_view_weights.reshape(b_micro,
                                                                                                      self.num_views, 1)
        image_feat_w = image_feat_w.reshape(-1, self.num_views * channel).type(self.dtype)

        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
        text_feat = text_feat.repeat(1, 1, self.num_views)

        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * torch.bmm(image_feat_w.unsqueeze(1), text_feat.transpose(1, 2)).squeeze(1)

        return logits


def extend_cfg(cfg):
    from yacs.config import CfgNode as CN
    cfg.MODEL.BACKBONE.NAME2 = "rn50"
    cfg.MODEL.PROJECT = CN()
    cfg.MODEL.PROJECT.NUM_VIEWS = 10


def reset_cfg(cfg, args):
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.seed:
        cfg.SEED = args.seed
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='output/sfda_teacher_scanobj_mcf')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--dataset-config-file', type=str, default='')
    parser.add_argument('--trainer', type=str, default='SFDA_PointCLIP')
    parser.add_argument('--backbone', type=str, default='RN50')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(args.config_file)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)

    if not cfg.OPTIM.LR:
        cfg.OPTIM.LR = 1e-4
    if not cfg.OPTIM.MAX_EPOCH:
        cfg.OPTIM.MAX_EPOCH = 15

    cfg.freeze()
    set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    trainer = build_trainer(cfg)
    if args.eval_only:
        trainer.test()
        return

    trainer.train()


if __name__ == '__main__':
    main()