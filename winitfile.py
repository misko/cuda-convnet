import numpy as n
from gpumodel import *
from scipy.io import loadmat

checkpoint_file = 'pretrained_models/imagenet'
layers_ = IGPUModel.load_checkpoint(checkpoint_file)['model_state']['layers']
layers = {}
for l in layers_:
  layers[l['name']] = l

def makew(name, idx, shape, params=None):
  layer_name = str(params[0])
  return layers[layer_name]['weights'][idx]

def makeb(name, shape, params=None):
  layer_name = str(params[0])
  return layers[layer_name]['biases']
