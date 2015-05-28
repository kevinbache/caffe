#!/bin/bash

./build/tools/caffe train \
  --solver=examples/mnist/mnist_500-300_sigmoid_softmax_solver_adam.prototxt
