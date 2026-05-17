# PointCLIP V2: Prompting CLIP and GPT for Powerful 3D Open-world Learning



## Introduction
PointCLIP V2 is a powerful 3D open-world learner, which improves the performance of PointCLIP with significant margins. V2 utilizes a realistic shape projection module for depth map generation, and adopts the LLM-assisted 3D prompt to align visual and language representations. Besides classification, PointCLIP V2 also conducts zero-shot part segmentation and 3D object detection.

## Environment
* conda create -n pointclip python=3.8 -y
* conda activate pointclip
* pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
* pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
* conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

* cd QSL/PointCLIP_V2-main/zeroshot_cls
* pip install -r requirements.txt
* pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
* pip install open3d
* cd Dassl3D
* pip install -e .

<!-- Examples of the synthesized depth map and attention map: -->
![Depth and Attention Map](figs/depth_attention_map.png)


<!-- The whole framework of PointCLIP V2: -->
<!-- ![Whole Framework](figs/whole_framework.png) -->


## Code

Please check the `zeroshot_cls` folder for [zero-shot 3D classification](https://github.com/yangyangyang127/PointCLIP_V2/tree/main/zeroshot_cls), and `zeroshot_seg` folder for [zero-shot part segmentation](https://github.com/yangyangyang127/PointCLIP_V2/tree/main/zeroshot_seg).

## Run
* source
* 先让 VLM 老师去熟悉 ShapeNet 的长相，生成在这个新数据集上的 2D 语义伪标签 
  python main_sfda.py `
    --config-file configs/trainers/PointCLIPV2_ZS/vit_b16.yaml `
    --dataset-config-file configs/datasets/pointda_shapenet.yaml `
    --output-dir output/sfda_teacher_M_to_S `
    --backbone ViT-B/16 `
    DATASET.ROOT datasets/PointDA_data_ply/shapenet `
    OPTIM.LR 0.0001 OPTIM.MAX_EPOCH 15
* 蒸馏学生模型（输入源模型与伪标签）
  python train_student.py `
    --config-file configs/trainers/PointCLIPV2_ZS/vit_b16.yaml `
    --dataset-config-file configs/datasets/pointda_shapenet.yaml `
    --output-dir output/train_student_M_to_S `
    --source-weight output/source_dgcnn_pointda10/dgcnn_source_best.pth `
    --teacher-weight output/sfda_teacher_M_to_S/sfda_mlp_epoch_15.pth `
    --epochs 150 `
    --batch-size 16 `
    --micro-batch 8 `
    --threshold 0.75 `
    DATASET.ROOT datasets/PointDA_data_ply/shapenet
* 循环自蒸馏 
  python train_self_distill.py `
    --config-file configs/trainers/PointCLIPV2_ZS/vit_b16.yaml `
    --dataset-config-file configs/datasets/pointda_shapenet.yaml `
    --output-dir output/self_distill_M_to_S `
    --teacher-weight output/train_student_M_to_S/best_student_dgcnn_finetuned.pth `
    --epochs 150 `
    --batch-size 16 `
    --micro-batch 8 `
    --threshold 0.80 `
    DATASET.ROOT datasets/PointDA_data_ply/shapenet