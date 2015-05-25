#!/bin/bash

./build/tools/caffe train \
  --solver=examples/mnist/lenet_solver_polyak.prototxt
