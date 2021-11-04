# Original-SRCNN

A modified version of morimoris's SRCNN implement based on the paper called *Image Super-Resolution Using Deep Convolutional Networks*.

## Overview

Now the program can output colorful images, but that don't mean this program can really process the color part of input images. In fact, I just keep the `Cr` and `Cb` color spaces and add them to the output images which only have `Y` color space to make them colorful. So, there is no essential difference between this program and the original one about the process of color spaces.

The processing effect is roughly as shown in the figure:

| <img src="https://raw.githubusercontent.com/9uanhuo/Original-SRCNN/main/images/low.jpg" alt="low" style="zoom:150%;" /> | <img src="https://raw.githubusercontent.com/9uanhuo/Original-SRCNN/main/images/pred.jpg" alt="pred" style="zoom:150%;" /> |
| :-------------------------: | :-------------------------: |
| Low Resolution              | SRCNN                       |

About the evaluation, SSIM is included now.

## Requirements

* cuda
* tensorflow
* numpy
* opencv-python

## Usage

The detail info about all the args are shown below. You can use the default value or use specified value. Please use `--mode` to specify a certain mode instead of using default `train_model` mode.
Generally speaking, the order in which the patterns are used is `train_data_create`, `test_data_create`, `train_model`, `evaluate`.
The code logic is pretty simple. If you have any confusion, you could check the code for details, or check the original author’s blog. Also an issue is welcomed :)

``` plaintext
usage: main.py [-h] [--train_height TRAIN_HEIGHT] [--train_width TRAIN_WIDTH] [--test_height TEST_HEIGHT]
               [--test_width TEST_WIDTH] [--train_dataset_num TRAIN_DATASET_NUM] [--test_dataset_num TEST_DATASET_NUM]
               [--train_cut_num TRAIN_CUT_NUM] [--test_cut_num TEST_CUT_NUM] [--train_path TRAIN_PATH] [--test_path TEST_PATH]
               [--learning_rate LEARNING_RATE] [--BATCH_SIZE BATCH_SIZE] [--EPOCHS EPOCHS] [--mode MODE]

Tensorflow SRCNN Example

optional arguments:
  -h, --help            show this help message and exit
  --train_height TRAIN_HEIGHT
                        Train data size(height)
  --train_width TRAIN_WIDTH
                        Train data size(width)
  --test_height TEST_HEIGHT
                        Test data size(height)
  --test_width TEST_WIDTH
                        Test data size(width)
  --train_dataset_num TRAIN_DATASET_NUM
                        Number of train datasets to generate
  --test_dataset_num TEST_DATASET_NUM
                        Number of test datasets to generate
  --train_cut_num TRAIN_CUT_NUM
                        Number of train data to be generated from a single image
  --test_cut_num TEST_CUT_NUM
                        Number of test data to be generated from a single image
  --train_path TRAIN_PATH
                        The path containing the train image
  --test_path TEST_PATH
                        The path containing the test image
  --learning_rate LEARNING_RATE
                        Learning_rate
  --BATCH_SIZE BATCH_SIZE
                        Training batch size
  --EPOCHS EPOCHS       Number of epochs to train for
  --mode MODE           train_data_create, test_data_create, train_model, evaluate
```

By the way, please don't forget to get the datasets and put them in a proper place before you run the code.

## Original author

All the changes have been authorized by the original author.

Original author's Blog：[DeepLearningを用いた超解像手法/SRCNNの実装](https://qiita.com/morimoris/items/b5d4966ce7c02b97ac98)

Original Repo：[morimoris/keras_SRCNN](https://github.com/morimoris/keras_SRCNN)

license：[morimoris/keras_SRCNN#4](https://github.com/morimoris/keras_SRCNN/issues/4)

## References

* [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092.pdf)

* [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

* [YCbCr](https://en.wikipedia.org/wiki/YCbCr)
