# Differentiable SVD


## Introduction

This repository contains:
1. The official Pytorch implementation of ICCV21 paper [Why Approximate Matrix Square Root Outperforms Accurate SVD in Global Covariance Pooling?](https://arxiv.org/abs/2105.02498)
2. The official Pytorch implementation of T-PAMI paper [On the Eigenvalues of Global Covariance Pooling for Fine-grained Visual Recognition](https://arxiv.org/pdf/2205.13282.pdf).
4. A collection of differentiable SVD methods utilized in our paper.

You can also find the presentation of our work via the [slides](https://docs.google.com/presentation/d/1qICV8bdQqogHsLIH0YJDsOOy9pG3OvLGdK9uCMXbHXQ/edit?usp=sharing) and via the [poster](https://github.com/KingJamesSong/DifferentiableSVD/blob/main/ICCV21_SVD_poster.pdf).

## About the paper

In this paper, we investigate the reason behind why approximate matrix square root calculated via Newton-Schulz iteration outperform the accurate ones computed by SVD from the perspectives of data precision and gradient smoothness. Various remedies for
computing smooth SVD gradients are investigated. We also propose a new spectral meta-layer that uses SVD in the forward pass, and Pad\'e approximants in the backward propagation to compute the gradients. The results of the so-called SVD-Pad\'e achieve state-of-the-art results on ImageNet and FGVC datasets.

## Differentiable SVD Methods
As the backward algorithm of SVD is prone to have numerical instability, we implement a variety of end-to-end SVD methods by manipulating the backward algortihms in this repository. They include:
- [***SVD-Pad\'e***](https://github.com/KingJamesSong/DifferentiableSVD/blob/main/src/representation/SVD_Pade.py): use Pad\'e approximants to closely approximate the gradient. It is proposed in our [ICCV21 paper](https://arxiv.org/abs/2105.02498). 
- [***SVD-Taylor***](https://github.com/KingJamesSong/DifferentiableSVD/blob/main/src/representation/SVD_Taylor.py): use Taylor polynomial to approximate the smooth gradient. It is proposed in our [ICCV21 paper](https://arxiv.org/abs/2105.02498) and the [TPAMI journal](https://arxiv.org/abs/2104.03821).
- [***SVD-PI***](https://github.com/KingJamesSong/DifferentiableSVD/blob/main/src/representation/SVD_PI.py): use Power Iteration (PI) to approximate the gradients. It is proposed in the [NeurIPS19 paper](https://arxiv.org/abs/1906.09023).
- [***SVD-Newton***](https://github.com/KingJamesSong/DifferentiableSVD/blob/main/src/representation/SVD_Newton.py): use the gradient of the Newton-Schulz iteration.
- [***SVD-Trunc***](https://github.com/KingJamesSong/DifferentiableSVD/blob/main/src/representation/SVD_Trunc.py): set a upper limit of the gradient and apply truncation.
- [***SVD-TopN***](https://github.com/KingJamesSong/DifferentiableSVD/blob/main/src/representation/SVD_TopN.py): select the Top-N eigenvalues and abandon the rest.
- [***SVD-Original***](https://github.com/KingJamesSong/DifferentiableSVD/blob/main/src/representation/SVD_Original.py): ordinary SVD with gradient overflow check.

In the task of global covaraince pooling, the SVD-Pad\'e achieves the best performances. You are free to try other methods in your research. 

## Implementation and Usage
The codes is modifed on the basis of [iSQRT-COV](https://github.com/jiangtaoxie/fast-MPN-COV).

See the [requirements.txt](https://github.com/KingJamesSong/DifferentiableSVD/blob/main/requirements.txt) for the specific required packages. 

To train AlexNet on ImageNet, choose a spectral meta-layer in the script and run:

`CUDA_VISIBLE_DEVICES=0,1 bash train_alexnet.sh`

The pre-trained models of ResNet-50 with SVD-Pad\'e is available via [Google Drive](https://drive.google.com/file/d/1ecVE3EklMgg0uwGTezvkvxDY6UxgEa95/view?usp=sharing). You can load the state dict by:

`model.load_state_dict(torch.load('pade_resnet50.pth.tar'))`


## Fine-grained Classification Usage (T-PAMI paper)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/on-the-eigenvalues-of-global-covariance/fine-grained-image-classification-on-stanford-1)](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford-1?p=on-the-eigenvalues-of-global-covariance)

The proposed Scaling Eigen Branch (SEB) is implemented [here](https://github.com/KingJamesSong/DifferentiableSVD/blob/main/src/representation/SEB.py), and the training scripts are available [here](https://github.com/KingJamesSong/DifferentiableSVD/tree/main/FGVC).

The pre-trained models are available via [Google Drive](https://drive.google.com/drive/folders/1yDvm8ZFcJTBtv0ezOg17EQ4o6OPQ2HM8?usp=sharing).

The perturbation-based visualization results are available via [Google Drive](https://drive.google.com/drive/folders/1PfrbJB05RA-__1k-tNI95-ljw__PiChE?usp=sharing), and the back-propagation-based visualizations are available via [Google Drive](https://drive.google.com/drive/folders/1rpd42JNPXUENCoO82Nk-0YtIY8BPvAeY?usp=sharing). We use [torchray](https://facebookresearch.github.io/TorchRay/) to generate backward-based explanations.

## Citation 
If you think the codes is helpful to your research, please consider citing our paper:

```
@inproceedings{song2021approximate,
  title={Why Approximate Matrix Square Root Outperforms Accurate SVD in Global Covariance Pooling?},
  author={Song, Yue and Sebe, Nicu and Wang, Wei},
  booktitle={ICCV},
  year={2021}
}
```

```
@article{song2022eigenvalues,
  title={On the Eigenvalues of Global Covariance Pooling for Fine-grained Visual Recognition},
  author={Song, Yue and Sebe, Nicu and Wang, Wei},
  journal={IEEE TPAMI},
  year={2022},
  publisher={IEEE}
}
```

## Contact

**If you have any questions or suggestions, please feel free to contact me**

`yue.song@unitn.it`
