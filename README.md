# Enhancing Multiscale Representations With Transformer for Remote Sensing Image Semantic Segmentation

## Introduction

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
conda create -n paddlevit python=3.7 -y
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


# Usage



*强调*  (示例：斜体)
 _强调_  (示例：斜体)
**加重强调**  (示例：粗体)
 __加重强调__ (示例：粗体)
***特别强调*** (示例：粗斜体)
___特别强调___  (示例：粗斜体)

* 1. 项目1  
* 2. 项目2  
* 3. 项目3  

代码
`<hello world>`  

`hello world`  

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
