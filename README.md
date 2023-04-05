# Enhancing Multiscale Representations With Transformer for Remote Sensing Image Semantic Segmentation

# Introduction

This repository is a PaddlePaddle implementation for our IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS) [[paper]](https://ieeexplore.ieee.org/document/10066301).


![14](https://user-images.githubusercontent.com/40911688/229135665-8b2e32e1-1880-48b2-a6eb-c60d0ef76905.png)

# Installation
This project uses PaddlePaddle. Go check them out if you don't have them locally installed.

### Environment Requirements
* Linux/MacOS/Windows  
* Python 3.6/3.7 
* PaddlePaddle 2.1.0+
* CUDA10.2+

> Note: It is recommended to install the latest version of PaddlePaddle to avoid some CUDA errors for PaddleViT training. For PaddlePaddle, please refer to this [link](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html) for stable version installation and this [link](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html#gpu) for develop version installation.

### Installation
1. Create a conda virtual environment and activate it.
```
conda create -n paddlevit python=3.8 -y
conda activate paddlevit
```
2. Install PaddlePaddle following the official instructions, e.g.,
```
conda install paddlepaddle-gpu==2.1.2 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```
> Note: please change the paddlepaddle version and cuda version accordingly to your environment.
3. Install dependency packages
  * General dependencies:
  ```
  pip install yacs pyyaml
  ```
### Reference Documents
If you have any questions about installing and using PaddlePaddle, please refer to [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) or [PaddleViT](https://github.com/BR-IDL/PaddleViT).

# Datasets Perparation
* You can download LoveDA dataset from [https://github.com/Junjue-Wang/LoveDA](https://github.com/Junjue-Wang/LoveDA).
* You can download Potsdam dataset from [https://isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx](https://isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx).
* You can download Vaihingen dataset from [https://isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx](https://isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx).

# Pretrained Weights Perparation
You can download the pre-trained weights you need in ".pdparams" format from [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg), [PaddleViT](https://github.com/BR-IDL/PaddleViT) or [PaddleClas](https://github.com/PaddlePaddle/PaddleClas). The weights of the backbone network used in the paper are provided here:

` BaiduYun:` [https://pan.baidu.com/s/1c5XCIDmYz9q5j0Nr9Ojbaw](https://pan.baidu.com/s/1c5XCIDmYz9q5j0Nr9Ojbaw)

` Password:` wcub

# Usage
We provide a simple demo to illustrate how to use EMRT for training and validation. Note that the method in this paper is run on a single GPU.
```
# Find the location of this project, for example
cd project/EMRT/semantic_segmentation/

Training:
# Modify the GPU number and yaml file path you want to use
CUDA_VISIBLE_DEVICES=0 python3 train.py --config ./configs/EMRT/EMRT_256x256_160k_potsdam.yaml

# Or just modify the GPU number you want to use if you define the default value of the "--config" parameter in train.py
CUDA_VISIBLE_DEVICES=0 python3 train.py

Validation:
# Use the same way to start validating
# CUDA_VISIBLE_DEVICES=0 python3 val.py --config ./configs/EMRT/EMRT_256x256_160k_potsdam.yaml --model_path ./EMRT_256x256_160k_potsdam_resnet50_pretrain_os32/best_model.pdparams

# Or just modify the GPU number you want to use if you define the default values ​​of the "--config" and "--model_path" parameters in val.py
CUDA_VISIBLE_DEVICES=0 python3 val.py
```

# Citation
If you find our work useful in your research, please consider citing:
```
@ARTICLE{10066301,
  author={Xiao, Tao and Liu, Yikun and Huang, Yuwen and Li, Mingsong and Yang, Gongping},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Enhancing Multiscale Representations With Transformer for Remote Sensing Image Semantic Segmentation}, 
  year={2023},
  volume={61},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2023.3256064}
}
```

# Other Links
* None
