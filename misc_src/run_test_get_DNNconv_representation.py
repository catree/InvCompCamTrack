import os
os.environ['THEANO_FLAGS'] = 'device=gpu1, floatX=float32' #, optimizer_including=cudnn
#!grep -h $(whoami) /tmp/lock-gpu*/info.txt 
# pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
# pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip


import lasagne
from lasagne.utils import floatX
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer # can be replaced with dnn layers
from lasagne.layers import BatchNormLayer, batch_norm, prelu
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer,DilatedConv2DLayer
from lasagne.layers import ConcatLayer, SliceLayer, ElemwiseMergeLayer
from lasagne.layers import ElemwiseSumLayer, standardize
from lasagne.layers import DenseLayer, ScaleLayer, PadLayer
from lasagne.nonlinearities import rectify, softmax, leaky_rectify
import theano
import theano.tensor as T

import numpy as np
import matplotlib.pyplot as plt
import io
from IPython.display import Image
import pickle
import scipy
from scipy import io 
from sklearn import metrics
from PIL import Image
import gzip
import os.path

input_var = T.ftensor4('x')

#define network
def define_vgg19_base(input_var=None, nonlinearity=rectify):
    net = {}
    net['input'] = InputLayer((None, 3, None, None), input_var=input_var)
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)

    layersstr = ['conv1_1','conv1_2',
                 'conv2_1','conv2_2',
                 'conv3_1','conv3_2','conv3_3','conv3_4',
                 'conv4_1','conv4_2','conv4_3','conv4_4',
                 'conv5_1','conv5_2','conv5_3','conv5_4']

    return net, layersstr
  
  
net, layersstr = define_vgg19_base(input_var=input_var)
                     

#https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl
model = pickle.load(open('/home/kroegert/Downloads/vgg19.pkl'))
mean_value = model['mean value']
lasagne.layers.set_all_param_values(net['conv5_4'], model['param values'][0:32])

# output function                    
outc_te = lasagne.layers.get_output([net['conv2_2'], net['conv3_4'],net['conv4_4'],net['conv5_4']], deterministic=True)
outc_fn = theano.function([input_var], outc_te)
                    
                                           
for all sequences:
  print seq
for  all iamges:
  filein = '/home/kroegert/Downloads/eiffelturm1.jpg'
  fileout = filein[:-4] + "_imgnet.pklz"
  
  if (not os.path.isfile(fileout)):
    img1 = plt.imread(filein).astype(float)
    img1 = np.swapaxes(np.swapaxes(img1, 1, 2), 0, 1)
    img1 = img1[::-1, :, :] - mean_value[:,None,None]
    img1 = (img1[np.newaxis]).astype(np.float32)
    out = outc_fn(img1)
    out = [np.swapaxes(np.swapaxes(x[0,:,:], 1, 0), 2, 1) for x in out]
    pickle.dump(out, gzip.open(fileout, 'w'), protocol=2)                                           

#import skimage.transform
#im = skimage.transform.resize(out[3], (img1.shape[2], img1.shape[3]), preserve_range=True)

#plt.ion()
#plt.figure()
#plt.imshow(out[3][:,:,1], interpolation='nearest')
#plt.show()

#plt.figure()
#plt.imshow(im[:,:,1], interpolation='nearest')
#plt.show()

