# test.CNN-F

Test the pre-trained CNN-F in TensorFlow/Pytorch with a simple classification model using MNIST.

Codes of CNN-F and pre-trained parameters are provided in [1].

Analyses of the weight files are contained in *test.cnnf.ipynb* and *cnnf.pytorch.ipynb*.

# Usage

## Tensorflow 1.12

- `python main.py`
- `tensorboard --logdir log`

## Pytorch

- `python main_torch.py`

## TensorFlow 2.1.0

- `python main_tf2.py`

# Data

MNIST, zooming into [224, 224, 3].

# Result

## tensorflow 1.12

- iter 0: 0.12269999995827675
- iter 450: 0.9907000076770782

![accuracy](accuracy.png)
![loss](loss.png)

## pytorch

- 1 epoch: 0.98

## tensorflow 2.1

- 1 epoch: 0.9742

# Environment

- tensorflow 1.12.0 / 2.1.0
- pytorch 1.4.0, torchvision 0.5.0
- cuda 9.0 / 10.1

# Pre-trained Weights

[cnnf-vggf](https://pan.baidu.com/s/1zxB_cHcalM8xbmauTS6_Xg#list/path=%2F)

# References

1. [jiangqy/DCMH-CVPR2017](https://github.com/jiangqy/DCMH-CVPR2017)
2. [tensorflow加载CNN-F/VGG-F预训练参数](https://blog.csdn.net/HackerTom/article/details/103189798)
