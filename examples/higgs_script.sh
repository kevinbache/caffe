#!/bin/bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
python hdf5_import_higgs.py
