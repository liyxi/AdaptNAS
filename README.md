# Adapting Neural Architectures Between Domains (AdaptNAS)

<p style="text-align: center;">

**Yanxi Li** <sup>1</sup>, **Zhaohui Yang** <sup>2, 3</sup>, **Yunhe Wang** <sup>2</sup>, **Chang Xu** <sup>1</sup>

<sup>1</sup> *School of Computer Science, University of Sydney, Australia*

<sup>2</sup> *Noah’s Ark Lab, Huawei Technologies, China*

<sup>3</sup> *Key Lab of Machine Perception (MOE), Department of Machine Intelligence, Peking University, China*

</p>

> A PyTorch implementation of ***Adapting Neural Architectures Between Domains*** (AdaptNAS) in NeurIPS 2020.
> 
> [[Abstract]](https://papers.nips.cc/paper/2020/hash/08f38e0434442128fab5ead6217ca759-Abstract.html) [[PDF]](https://papers.nips.cc/paper/2020/file/08f38e0434442128fab5ead6217ca759-Paper.pdf)
>
> This code is based on [**DARTS**](https://github.com/quark0/darts) (https://github.com/quark0/darts).

## Overview

Neural architecture search (NAS) has demonstrated impressive performance in automatically designing high-performance neural networks. The power of deep neural networks is to be unleashed for analyzing a large volume of data (e.g. ImageNet), but the architecture search is often executed on another smaller dataset (e.g. CIFAR-10) to finish it in a feasible time. However, it is hard to guarantee that the optimal architecture derived on the proxy task could maintain its advantages on another more challenging dataset. This paper aims to improve the generalization of neural architectures via domain adaptation. We analyze the generalization bounds of the derived architecture and suggest its close relations with the validation error and the data distribution distance on both domains. These theoretical analyses lead to AdaptNAS, a novel and principled approach to adapt neural architectures between domains in NAS. Our experimental evaluation shows that only a small part of ImageNet will be sufficient for AdaptNAS to extend its architecture success to the entire ImageNet and outperform state-of-the-art comparison algorithms.

## Requirements

This project is developed and tested under the following environment:

- python>=3.7.5
- pytorch>=1.3.1
- torchvision>=0.4.2

## Instruction

### Search

> Seach code will be available soon.

### Retrain

To retrain the searched architecture on the ImageNet, execute the following command:
```bash
python train_imagenet.py \
    --arch arch_AdaptNAS \
    --batch_size 512 \
    --learning_rate 0.25 \
    --auxiliary \
    --epochs 250 \
    --num_workers 48
```

We use four Tesla V100 GPUs for retraining. If you use a different number of GPUs, the `bach_size` and `learning_rate` should be adjusted correspondingly.

## Citation
```
@inproceedings{li2020adapting,
    title={Adapting Neural Architectures Between Domains},
    author={Li, Yanxi and Yang, Zhaohui and Wang, Yunhe and Xu, Chang},
    booktitle={Advances in Neural Information Processing Systems},
    year={2020}
}
```
