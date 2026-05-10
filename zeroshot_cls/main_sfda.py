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
from dynamic_prompt import TopologicalViewRouter

import datasets.pointda


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
        self.num_views = getattr(cfg.MODEL.PROJECT, 'NUM_VIEWS', 10)

        # 多提示词集成 (Prompt Ensembling)
        classnames = self.dm.dataset.classnames
        templates = [
            "a depth map of a {}.",
            "a 3d point cloud scan of a {}.",
            "a noisy 3d model of a {}."
        ]
        print(f"\n>>> [Oracle 初始化] 正在使用 {len(templates)} 种混合模板生成强化文本基准...")

        with torch.no_grad():
            text_features = []
            for classname in classnames:
                texts = [t.format(classname) for t in templates]
                tokens = clip.tokenize(texts).to(device)
                class_embeddings = clip_model.encode_text(tokens)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                text_features.append(class_embedding)
            self.fixed_text_features = torch.stack(text_features).type(clip_model.dtype)

        self.view_router = TopologicalViewRouter(geo_dim=16, num_views=self.num_views).to(device)

        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.model = self.view_router

        self.optim = torch.optim.AdamW(
            self.view_router.parameters(),
            lr=1e-4,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=cfg.OPTIM.MAX_EPOCH)

    def train(self):
        print(f"\n>>> [Oracle Check] 混合提示词下初始 10 类 Zero-Shot 性能摸底...")
        self.test()

        print("\n>>> 开始论文终极形态：Oracle-Guided Topological View Routing (神谕引导 TVR)...")
        target_loader = self.train_loader_x

        for epoch in range(self.cfg.OPTIM.MAX_EPOCH):
            self.view_router.train()
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
                f"Epoch {epoch + 1} 结束. Avg Loss: {total_loss / len(target_loader):.4f}, Oracle 高自信比例: {avg_mask:.1%}")

            self.test()

            save_path = os.path.join(self.cfg.OUTPUT_DIR, f"sfda_mlp_epoch_{epoch + 1}.pth")
            torch.save(self.view_router.state_dict(), save_path)

    def forward_backward(self, batch):
        pc = batch["img"].cuda().float()
        if pc.dim() == 4:
            pc = pc.squeeze(1)

        B = pc.shape[0]
        micro_batch_size = 8
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

                image_feat_views = image_feat.reshape(b_micro, self.num_views, channel)

                oracle_img_feat = image_feat_views.mean(dim=1)
                oracle_img_feat = oracle_img_feat / (oracle_img_feat.norm(dim=-1, keepdim=True) + 1e-6)

                logit_scale = self.clip_model.logit_scale.exp()
                oracle_logits = logit_scale * torch.matmul(oracle_img_feat, self.fixed_text_features.T)

                oracle_probs = F.softmax(oracle_logits, dim=1)
                oracle_conf, oracle_preds = torch.max(oracle_probs, dim=1)

                # 🚨 微调 1: 降低神谕门槛到 0.45，释放更多“真金”锚点 (约 15%~20%)
                mask = (oracle_conf > 0.45).float()

            dynamic_weights = self.view_router(pc_micro)

            routed_img_feat = torch.bmm(dynamic_weights.unsqueeze(1), image_feat_views).squeeze(1)
            routed_img_feat = routed_img_feat / (routed_img_feat.norm(dim=-1, keepdim=True) + 1e-6)

            routed_logits = logit_scale * torch.matmul(routed_img_feat, self.fixed_text_features.T)

            ce_loss = F.cross_entropy(routed_logits, oracle_preds, reduction='none')
            masked_ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)

            routed_probs = F.softmax(routed_logits, dim=1)
            ent_loss = -torch.sum(routed_probs * torch.log(routed_probs + 1e-8), dim=1)
            masked_ent_loss = (ent_loss * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)

            # 🚨 微调 2: 大幅削弱 Entropy Loss (0.1)，防止其喧宾夺主导致后期坍塌
            if mask.sum() > 0:
                loss = 1.0 * masked_ce_loss + 0.1 * masked_ent_loss
            else:
                loss = 0.1 * masked_ent_loss

            loss.backward()
            total_loss_val += loss.item() * (b_micro / B)
            total_mask_val += (mask.sum().item() / b_micro) * (b_micro / B)

        torch.nn.utils.clip_grad_norm_(self.view_router.parameters(), max_norm=1.0)
        self.optim.step()

        return {
            "loss": total_loss_val,
            "mask_ratio": total_mask_val
        }

    def model_inference(self, pc, label=None):
        if pc.dim() == 4:
            pc = pc.squeeze(1)

        dynamic_weights = self.view_router(pc)

        images = self.pc_views.get_img(pc).cuda()
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=True)
        images = images.type(self.dtype)

        image_feat = self.clip_model.visual(images)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        channel = image_feat.shape[-1]
        b_micro = pc.shape[0]
        image_feat_views = image_feat.reshape(b_micro, self.num_views, channel)

        routed_img_feat = torch.bmm(dynamic_weights.unsqueeze(1), image_feat_views).squeeze(1)
        routed_img_feat = routed_img_feat / (routed_img_feat.norm(dim=-1, keepdim=True) + 1e-6)

        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * torch.matmul(routed_img_feat, self.fixed_text_features.T)

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
    parser.add_argument('--output-dir', type=str, default='output/sfda_teacher_pointda')
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
    if args.opts:
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