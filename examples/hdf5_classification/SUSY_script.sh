#!/usr/bin/env sh

./build/tools/caffe train \
  --solver=examples/hdf5_classification/susy_solver.prototxt
