# test.CNN-F

Test the pre-trained CNN-F in TensorFlow/Pytorch with a simple classification model using MNIST.

Codes of CNN-F and pre-trained parameters are provided in [1].

# Usage

## Tensorflow

- `python main.py`
- `tensorboard --logdir log`

## Pytorch


# Data

MNIST, zooming into [224, 224, 3].

# Result

## tensorflow

- iter 0: 0.12269999995827675
- iter 450: 0.9907000076770782

# pytorch

- epoch 

![accuracy](accuracy.png)
![loss](loss.png)

# Environment

- tensorflow 1.12.0
- pytorch 1.4.0
- cuda 9.0

# Pre-trained Weights

[cnnf-vggf](https://pan.baidu.com/s/1zxB_cHcalM8xbmauTS6_Xg#list/path=%2F)

# References

1. [jiangqy/DCMH-CVPR2017](https://github.com/jiangqy/DCMH-CVPR2017)
2. [tensorflow加载CNN-F/VGG-F预训练参数](https://blog.csdn.net/HackerTom/article/details/103189798)
