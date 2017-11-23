#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os
import math

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

"""
CIFAR10 DenseNet example. See: https://arxiv.org/abs/1610.02915
Code is developed based on Yuxin Wu's ResNet implementation: https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet
"""
# TODO: update runtime and actual performance

BATCH_SIZE = 64


class Model(ModelDesc):
  def __init__(self, depth):
    super(Model, self).__init__()
    self.N = int((depth - 4) / 3)
    self.growthRate = 12

  def _get_inputs(self):
    return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
            InputDesc(tf.int32, [None], 'label')
            ]

  def _build_graph(self, input_vars):
    image, label = input_vars
    image = image / 128.0 - 1

    def conv(name, l, channels, kernel=3, stride=1):
      # TODO: update weight init
      return Conv2D(name, l, channels, kernel, stride=stride,
                    nl=tf.identity, use_bias=False,
                    W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / channels)))

    def add_layer(name, input_l, final_channels, stride):
      shape = input_l.get_shape().as_list()
      final_channels = math.round(final_channels)

      with tf.variable_scope(name) as scope:
        c = BatchNorm('bn0', input_l)

        c = conv('conv1', c, final_channels, 1, 1)
        c = BatchNorm('bn1', c)
        c = tf.nn.relu(c)

        c = conv('conv2', c, final_channels, 3, stride)
        c = BatchNorm('bn2', c)
        c = tf.nn.relu(c)

        c = conv('conv3', c, final_channels * 4, 1, 1)
        c = BatchNorm('bn3', c)

        # pad residual with zero channels
        # TODO: check with ResNet if they pad with 75% zeros on first bottleneck block
        shape[-2] = final_channels - shape[-2]
        residual_padding = tf.zeros(shape, tf.float32)
        residual = tf.concat((input_l, residual_padding), 2)
        # reduce map
        if stride == 0:
          residual = AvgPooling('pool', residual, 2)
        # add
        l = tf.add_n(c, residual)
      return l

    def pyramid_net(name):
      n = (depth - 2)/9   # -2 transitional layers, /3 blocks, /3 bottleneck layers in each pyramid-net
      add_channels = alpha / (3 * n)

      l = conv('conv0', image, 16, 1)
      l = BatchNorm('bn1', l)

      for i, block in enumerate(['block1', 'block2', 'block3']):
        with tf.variable_scope('block') as scope:
          for j in range(n):
            stride = 1 if i == 0 or j > 0 else 2  # apply stride=2 exactly twice between 1st-2nd and 2nd-3rd blocks
            l = add_layer('block%d_%d' % (i+1, j), l, add_channels, stride)

      l = BatchNorm('bnlast', l)
      l = tf.nn.relu(l)
      l = GlobalAvgPooling('gap', l)
      logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)

      return logits

    logits = pyramid_net("dense_net")

    prob = tf.nn.softmax(logits, name='output')

    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
    cost = tf.reduce_mean(cost, name='cross_entropy_loss')

    wrong = prediction_incorrect(logits, label)
    # monitor training error
    add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

    # weight decay on all W
    wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
    add_moving_summary(cost, wd_cost)

    add_param_summary(('.*/W', ['histogram']))  # monitor W
    self.cost = tf.add_n([cost, wd_cost], name='cost')

  def _get_optimizer(self):
    lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
    tf.summary.scalar('learning_rate', lr)
    return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test):
  isTrain = train_or_test == 'train'
  ds = dataset.Cifar10(train_or_test)
  pp_mean = ds.get_per_pixel_mean()
  if isTrain:
    augmentors = [
      imgaug.CenterPaste((40, 40)),
      imgaug.RandomCrop((32, 32)),
      imgaug.Flip(horiz=True),
      imgaug.MapImage(lambda x: x - pp_mean),
    ]
  else:
    augmentors = [
      imgaug.MapImage(lambda x: x - pp_mean)
    ]
  ds = AugmentImageComponent(ds, augmentors)
  ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
  if isTrain:
    ds = PrefetchData(ds, 3, 2)
  return ds


def get_config():
  log_dir = 'train_log/cifar10-single-fisrt%s-second%s-max%s' % (
  str(args.drop_1), str(args.drop_2), str(args.max_epoch))
  logger.set_logger_dir(log_dir, action='n')

  # prepare dataset
  dataset_train = get_data('train')
  steps_per_epoch = dataset_train.size()
  dataset_test = get_data('test')

  return TrainConfig(
    dataflow=dataset_train,
    callbacks=[
      ModelSaver(),
      InferenceRunner(dataset_test,
                      [ScalarStats('cost'), ClassificationError()]),
      ScheduledHyperParamSetter('learning_rate',
                                [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)])
    ],
    model=Model(depth=args.depth),
    steps_per_epoch=steps_per_epoch,
    max_epoch=args.max_epoch,
  )


if __name__ == '__main__':
  # TODO: verify hyper parameters with the paper
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')  # nargs='*' in multi mode
  parser.add_argument('--load', help='load model')
  parser.add_argument('--drop_1', default=150, help='Epoch to drop learning rate to 0.01.')  # nargs='*' in multi mode
  parser.add_argument('--drop_2', default=225, help='Epoch to drop learning rate to 0.001')
  parser.add_argument('--depth', default=40, help='The depth of densenet')
  parser.add_argument('--max_epoch', default=300, help='max epoch')
  args = parser.parse_args()

  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  config = get_config()
  if args.load:
    config.session_init = SaverRestore(args.load)
  if args.gpu:
    config.nr_tower = len(args.gpu.split(','))
  SyncMultiGPUTrainer(config).train()