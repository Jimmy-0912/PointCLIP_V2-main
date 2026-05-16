# PointCLIP V2: Prompting CLIP and GPT for Powerful 3D Open-world Learning

Official implementation of [PointCLIP V2: Prompting CLIP and GPT for Powerful 3D Open-world Learning](https://arxiv.org/abs/2211.11682).

The V1 version of [PointCLIP](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_PointCLIP_Point_Cloud_Understanding_by_CLIP_CVPR_2022_paper.pdf) accepted by CVPR 2022 is open-sourced at [here](https://github.com/ZrrSkywalker/PointCLIP).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pointclip-v2-adapting-clip-for-powerful-3d/zero-shot-transfer-3d-point-cloud-2)](https://paperswithcode.com/sota/zero-shot-transfer-3d-point-cloud-2?p=pointclip-v2-adapting-clip-for-powerful-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pointclip-v2-adapting-clip-for-powerful-3d/training-free-3d-point-cloud-classification-1)](https://paperswithcode.com/sota/training-free-3d-point-cloud-classification-1?p=pointclip-v2-adapting-clip-for-powerful-3d)

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

## Introduction
PointCLIP V2 is a powerful 3D open-world learner, which improves the performance of PointCLIP with significant margins. V2 utilizes a realistic shape projection module for depth map generation, and adopts the LLM-assisted 3D prompt to align visual and language representations. Besides classification, PointCLIP V2 also conducts zero-shot part segmentation and 3D object detection.


<!-- Examples of the synthesized depth map and attention map: -->
![Depth and Attention Map](figs/depth_attention_map.png)


<!-- The whole framework of PointCLIP V2: -->
<!-- ![Whole Framework](figs/whole_framework.png) -->


## Code

Please check the `zeroshot_cls` folder for [zero-shot 3D classification](https://github.com/yangyangyang127/PointCLIP_V2/tree/main/zeroshot_cls), and `zeroshot_seg` folder for [zero-shot part segmentation](https://github.com/yangyangyang127/PointCLIP_V2/tree/main/zeroshot_seg).

## Contributors
[Xiangyang Zhu](https://github.com/yangyangyang127), [Renrui Zhang](https://github.com/ZrrSkywalker)


## Citation
Thanks for citing our paper:

```
@article{Zhu2022PointCLIPV2,
    title={PointCLIP V2: Prompting CLIP and GPT for Powerful 3D Open-world Learning},
    author={Zhu, Xiangyang and Zhang, Renrui and He, Bowei and Guo, Ziyu and Zeng, Ziyao and Qin, Zipeng and Zhang, Shanghang and Gao, Peng},
    journal={arXiv preprint arXiv:2211.11682},
    year={2022},
}
```

## Contact
If you have any question about this project, please feel free to contact xiangyzhu6-c@my.cityu.edu.hk and zhangrenrui@pjlab.org.cn.

