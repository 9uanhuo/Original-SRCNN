# Original-SRCNN

## Overview

A modified version of morimoris's SRCNN implement based on the paper called *Image Super-Resolution Using Deep Convolutional Networks*.

Now the program can output colorful images, but that's not mean this program can really process the color part of input images. In fact, I just keep the `Cr` and `Cb` color spaces and add them to the output images which only have `Y` color space to make them colorful. So, there is no essential difference between this program and the original one.

The processing effect is roughly as shown in the figure:

| <img src="https://raw.githubusercontent.com/9uanhuo/Original-SRCNN/main/images/low.jpg" alt="low" style="zoom:150%;" /> | <img src="https://raw.githubusercontent.com/9uanhuo/Original-SRCNN/main/images/pred.jpg" alt="pred" style="zoom:150%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Low Resolution                                               | SRCNN                                                        |

## Requirements

* cuda
* tensorflow
* numpy
* opencv-python

## Original author

All the changes have been authorized by the original author.

Original author's Blog：[DeepLearningを用いた超解像手法/SRCNNの実装](https://qiita.com/morimoris/items/b5d4966ce7c02b97ac98)

Original Repo：[morimoris/keras_SRCNN](https://github.com/morimoris/keras_SRCNN)

license：[morimoris/keras_SRCNN#4](https://github.com/morimoris/keras_SRCNN/issues/4)

## References

* [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092.pdf)

* [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

* [YCbCr](https://en.wikipedia.org/wiki/YCbCr)

