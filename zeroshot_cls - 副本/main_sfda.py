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

        # 智能解析 Backbone 名字，支持 RN50 和 ViT 自由切换
        dataset_name = cfg.DATASET.NAME.lower()
        raw_name = cfg.MODEL.BACKBONE.NAME.lower().replace('-', '_').replace('/', '')
        bb_name = getattr(cfg.MODEL.BACKBONE, 'NAME2', raw_name)
        if bb_name == "rn50" and "vit" in raw_name:
            bb_name = raw_name  # 覆盖掉写死的 rn50

        prompt_key = f"{dataset_name}_{bb_name}_test_prompts"

        if prompt_key in best_prompt_weight:
            text_prompts = best_prompt_weight[prompt_key]
        else:
            classnames = self.dm.dataset.classnames
            text_prompts = [f"a point cloud of a {name}." for name in classnames]

        self.prompt_learner = DynamicPromptLearner(
            clip_model, text_prompts, ctx_length=4, geo_dim=16
        ).to(device)

        weight_key = f"{dataset_name}_{bb_name}_test_weights"
        if weight_key in best_prompt_weight:
            self.view_weights = torch.Tensor(best_prompt_weight[weight_key]).to(device)
        else:
            self.view_weights = torch.ones(10).to(device)

        self.num_views = getattr(cfg.MODEL.PROJECT, 'NUM_VIEWS', 10)
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.model = self.prompt_learner

        # 🚨 解除封印：恢复使用命令行的学习率 🚨
        self.optim = torch.optim.AdamW(
            self.prompt_learner.meta_net.parameters(),
            lr=cfg.OPTIM.LR,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=cfg.OPTIM.MAX_EPOCH)

    def train(self):
        print(f"\n>>> [Sanity Check] {self.cfg.MODEL.BACKBONE.NAME} 初始 Zero-Shot 性能摸底...")
        self.test()

        print("\n>>> 开始进入原生 SFDA 训练循环 (动态适应中)...")
        target_loader = self.train_loader_x

        for epoch in range(self.cfg.OPTIM.MAX_EPOCH):
            self.prompt_learner.train()
            total_loss = 0.0

            pbar = tqdm(target_loader, desc=f"Epoch {epoch + 1}/{self.cfg.OPTIM.MAX_EPOCH}")

            for batch_idx, batch in enumerate(pbar):
                loss_summary = self.forward_backward(batch)
                total_loss += loss_summary["loss"]
                pbar.set_postfix(loss_summary)

            self.sched.step()
            print(f"Epoch {epoch + 1} finished. Avg Loss: {total_loss / len(target_loader):.4f}")

            self.test()

            save_path = os.path.join(self.cfg.OUTPUT_DIR, f"sfda_mlp_epoch_{epoch + 1}.pth")
            torch.save(self.prompt_learner.meta_net.state_dict(), save_path)

    def forward_backward(self, batch):
        pc = batch["img"].cuda().float()
        if pc.dim() == 4:
            pc = pc.squeeze(1)

        B = pc.shape[0]
        micro_batch_size = 4

        self.optim.zero_grad()

        total_loss_val = 0.0
        total_ent_val = 0.0
        total_kl_val = 0.0

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
                image_feat_w = image_feat.reshape(-1, self.num_views, channel) * self.view_weights.reshape(1, -1, 1)
                image_feat_w = image_feat_w.reshape(-1, self.num_views * channel).type(self.dtype)

                text_feat_zs = self.prompt_learner.forward_zs(b_micro)
                text_feat_zs = text_feat_zs / (text_feat_zs.norm(dim=-1, keepdim=True) + 1e-6)
                text_feat_zs = text_feat_zs.repeat(1, 1, self.num_views)
                logits_zs = self.clip_model.logit_scale.exp() * torch.bmm(image_feat_w.unsqueeze(1),
                                                                          text_feat_zs.transpose(1, 2)).squeeze(1)

            text_feat = self.prompt_learner(pc_micro)
            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
            text_feat = text_feat.repeat(1, 1, self.num_views)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * torch.bmm(image_feat_w.unsqueeze(1), text_feat.transpose(1, 2)).squeeze(1)

            probs = F.softmax(logits, dim=1)

            kl_loss = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits_zs.detach(), dim=1),
                               reduction='batchmean')
            entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            mean_probs = torch.mean(probs, dim=0)
            diversity_loss = torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            # 🚨 解除封印：调低 KL 约束，增加 MLP 自我学习的比重 🚨
            loss = (0.5 * kl_loss + 1.0 * entropy_loss + 0.5 * diversity_loss) * (b_micro / B)
            loss.backward()

            total_loss_val += loss.item()
            total_ent_val += entropy_loss.item() * (b_micro / B)
            total_kl_val += kl_loss.item() * (b_micro / B)

        torch.nn.utils.clip_grad_norm_(self.prompt_learner.parameters(), max_norm=1.0)
        self.optim.step()

        return {
            "loss": total_loss_val,
            "kl_loss": total_kl_val,
            "ent_loss": total_ent_val
        }

    def model_inference(self, pc, label=None):
        if pc.dim() == 4:
            pc = pc.squeeze(1)

        images = self.pc_views.get_img(pc).cuda()
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=True)
        images = images.type(self.dtype)

        image_feat = self.clip_model.visual(images)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        channel = image_feat.shape[-1]
        image_feat_w = image_feat.reshape(-1, self.num_views, channel) * self.view_weights.reshape(1, -1, 1)
        image_feat_w = image_feat_w.reshape(-1, self.num_views * channel).type(self.dtype)

        text_feat = self.prompt_learner(pc)
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
    parser.add_argument('--output-dir', type=str, default='output/sfda_run')
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