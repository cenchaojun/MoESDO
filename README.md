<div align = 'center'>
<h1>MoESOD</h1>
<h3>A multi-scale vision mixture-of-experts for salient object detection with Kolmogorov–Arnold adapter</h3>
Chaojun Cen, Fei Li, Ping Hu, Zhenbo Li

China Agricultural University


</div>


## Abstract
Diverse domains and object variations make salient object detection (SOD) a challenging task in computer vision. Many previous studies have adopted multi-scale neural networks with attention mechanisms. Although they are popular, the design of their networks lacks sufficient flexibility, which hinders their generalization across objects of different scales and domains. To address the above issue, we propose a novel mixture-of-experts salient object detection (MoESOD) approach. We design a multi-scale mixture-of-experts (MMoE) module, essentially large neural networks, to improve the model’s expressive power and generalization ability without significantly increasing computational cost. By leveraging expert competition and collaboration strategies, the MMoE module effectively integrates contributions from different experts. The MMoE module not only captures multi-scale features but also effectively fuses semantic information across scales through the expert gating mechanism. Additionally, the novel kolmogorov-arnold network adapter (KANA) is designed to enhance the model’s flexibility, allowing it to adapt easily to SOD tasks across different domains. Comprehensive experiments show that MoESOD consistently achieves higher performance than, or at least comparable performance to, state-of-the-art methods on 10 different SOD benchmarks and 2 downstream tasks. To the best of our knowledge, this is the first study to explore kolmogorov-arnold network within the SOD community.
## Overview
<img width="4309" height="1810" alt="pipeline" src="https://github.com/user-attachments/assets/be6869bb-9dc3-4f4a-9ab8-c724d333be4c" />

<p align="center">
 
</p>

## Usage

### Installation

#### Step 1:
Clone this repository

```bash
https://github.com/cenchaojun/MoESOD.git
cd MoESOD
```

#### Step 2:

##### Create a new conda environment

```bash
conda create --name moesod python=3.9
conda activate moesod
```

##### Install Dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu<your cuda version>
pip install -r requirements.txt
```

##### Set up Datasets
````
-- SOD
    |-- DUTS
    |   |-- train
    |   |   |-- image
    |   |   |-- mask
    |   |-- test
    |   |   |-- image
    |   |   |-- mask
    |-- DUT-OMRON
    |   |-- test
    |   |   |-- image
    |   |   |-- mask
    |-- HKU-IS
    |   |-- test
    |   |   |-- image
    |   |   |-- mask
    |-- ECSSD
    |   |-- test
    |   |   |-- image
    |   |   |-- mask
    |-- PASCAL-S
    |   |-- test
    |   |   |-- image
    |   |   |-- mask
````
You can either go to [DUTS](http://saliencydetection.net/duts/), [DUT-OMRON](http://saliencydetection.net/dut-omron/), [HKU-IS](https://i.cs.hku.hk/~yzyu/research/deep_saliency.html), [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), and [PASCAL-S](https://cbs.ic.gatech.edu/salobj/) respectively to download and configure files, or you can directly download [SOD GoogleDrive](https://drive.google.com/file/d/1xY1nB1KMUNXYV0CyKTgYoYudekGFUp3R/view?usp=drive_link) that I have already configured.
Download [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).
##### Train

example train.sh when you load previous checkpoint
```
CUDA_VISIBLE_DEVICES=<gpu device> python -m torch.distributed.launch --nproc_per_node=<gpu num> train.py \
    --batch_size 50 \
    --num_workers 2 \
    --data_path <path>/SOD/DUTS \
    --resume <path>/model_epoch<epoch id>.pth
    --img_size 384
```

##### Evaluation


```
python test.py \
    --data_path <path>/SOD \
    --img_size 384 \
    --checkpoint <path>/<checkpoint>.pth \
    --gpu_id 0 \
    --result_path <output>/<path>
```


## Citation

```
@article{cen2025multi,
  title={A multi-scale vision mixture-of-experts for salient object detection with Kolmogorov-Arnold adapter},
  author={Cen, Chaojun and Li, Fei and Hu, Ping and Li, Zhenbo},
  journal={Neurocomputing},
  pages={130349},
  year={2025},
  publisher={Elsevier}
}
```
## Acknowledgments
Our project is based on [MDSAM](https://github.com/BellyBeauty/MDSAM). Thanks for their awesome works.
