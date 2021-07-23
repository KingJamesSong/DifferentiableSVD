# Differentiable SVD


## Introduction

This repository contains:
1. The official Pytorch implementation of ICCV21 paper [Why Approximate Matrix Square Root Outperforms Accurate SVD in Global Covariance Pooling?](https://arxiv.org/abs/2105.02498)
2. A collection of differentiable SVD methods.

## About the paper

In this paper, we investigate the reason behind why approximate matrix square root calculated via Newton-Schulz iteration outperform the accurate ones computed by SVD from the perspectives of data precision and gradient smoothness. Various remedies for
computing smooth SVD gradients are investigated. We also propose a new GCP meta-layer that uses SVD in the forward pass, and Pad\'e approximants in the backward propagation to compute the gradients. The results of the so-called SVD-Pad\'e achieve state-of-the-art results.

## Differentiable SVD Methods



## Citation 
If you think the codes is helpful to your research, please consider cite our paper:

         @article{song2021approximate,
                  title={Why Approximate Matrix Square Root Outperforms Accurate SVD in Global Covariance Pooling?},
                  author={Song, Yue and Sebe, Nicu and Wang, Wei},
                  journal={arXiv preprint arXiv:2105.02498},
                  year={2021}
          }
## Contact

**If you have any questions or suggestions, please contact me**

`yue.song@unitn.it`
