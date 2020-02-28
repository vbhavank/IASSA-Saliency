# IASSA-Saliency
Source code for our WACV 2020 paper ``Iterative and Adaptive Sampling with Spatial Attention for Black-Box Model Explanations" https://arxiv.org/abs/1912.08387

## Abstract
######  &nbsp;&nbsp;&nbsp;  Deep neural networks have achieved great success in many real-world applications, yet it remains unclear and difficult to explain their decision-making process to an end-user. In this paper, we address the explainable AI problem for deep neural networks with our proposed framework, named IASSA, which generates an importance map indicating how salient each pixel is for the model's prediction with an iterative and adaptive sampling module. We employ an affinity matrix calculated on multi-level deep learning features to explore long-range pixel-to-pixel correlation, which can shift the saliency values guided by our long-range and parameter-free spatial attention. Extensive experiments on the MS-COCO dataset show that our proposed approach matches or exceeds the performance of state-of-the-art black-box explanation methods.

![alt text](https://github.com/vbhavank/IASSA-Saliency/blob/master/imgs/block_diagram.png "Block diagram")

###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure: The framework of our unsupervised saliency map extraction method for an explanation of a black-box model. 

#### To run Iterative and Adaptive Sampling to explain a black box model run:
```python
python iassa.py --query_path "/PATH/TO/INPUT_IMAGES/*.IMAGE_FORMAT" --save_path /PATH/TO/SAVE/SALIENCY/MAPS/ --gt_path /PATH/TO/FOLDER/WITH/GROUNDTRUTH/ANNOTATIONS/ 
```
### Reference
```
@InProceedings{Vasu_2020_WACV,
author = {Vasu, Bhavan and Long, Chengjiang},
title = {Iterative and Adaptive Sampling with Spatial Attention for Black-Box Model Explanations},
booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
month = {March},
year = {2020}
}```
