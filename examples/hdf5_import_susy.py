import numpy as np
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import os
import h5py
import shutil
import tempfile
import sklearn
import gzip
import sklearn.linear_model
f = gzip.open('SUSY.csv.gz', 'rb')
shape=(5000000,19)
X=np.zeros(shape)
for i in range(shape[0]):
    X[i,:]=np.array(map(float, f.readline().split(',')))
f.close()
y=X[:,0]
X=np.delete(X,0,1)
# Split into train and test
X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y)
print('Type of x is',type(X))
print('Type of y is',type(y))
print('Shape of X is ',X.shape)
print('Shape of y is',y.shape)
print('Shape of Xt (Xtest) is ',Xt.shape)
print('Shape of yt (Ytest) is',yt.shape)
dirname = os.path.abspath('./hdf5_classification/data')
if not os.path.exists(dirname):
    os.makedirs(dirname)
train_filename = os.path.join(dirname, 'train_SUSY.h5')
test_filename = os.path.join(dirname, 'test_SUSY.h5')
# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
with h5py.File(train_filename, 'w') as f:
    f['data'] = X
    f['label'] = y
with open(os.path.join(dirname, 'train_SUSY.txt'), 'w') as f:
    f.write(train_filename + '\n')
    f.write(train_filename + '\n')
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File(test_filename, 'w') as f:
    f.create_dataset('data', data=Xt, **comp_kwargs)
    f.create_dataset('label', data=yt, **comp_kwargs)
with open(os.path.join(dirname, 'test_SUSY.txt'), 'w') as f:
    f.write(test_filename + '\n')
