import numpy as np
import os
import cPickle
import random
import cv2
from utils.cython_bbox import bbox_overlaps
from anchor import get_anchors
import numpy.random as npr
from bbox_transform import bbox_transform
import find_mxnet
import mxnet as mx
from resnet import get_symbol_data
import mxnet.optimizer as opt
from mxnet.optimizer import get_updater
import logging
import time

def get_symbol():
    data = mx.sym.var(name='data')
    resnet101 = get_symbol_data(data, num_classes=1000, num_layers=101, image_shape=(3,500,500))
    internals = resnet101.get_internals()
    resnet101_22b = internals['_plus29_output']
    resnet101_3c = internals['_plus6_output']

    score_res4 = mx.sym.Convolution(data=resnet101_22b, num_filter=150, kernel=(1,1), stride=(1,1), pad=(0,0), name='score_res4')
    bilinear_weight = mx.sym.Variable(name='bilinear_weight', init=mx.init.Bilinear(), attr={'lr_mult':'0.0'})
    score4_up = mx.sym.UpSampling(*[score_res4, bilinear_weight], num_filter=150, scale=2, sample_type='bilinear',name='score4_up', num_args=2)
    score4 = mx.sym.Crop(*[score4_up, resnet101_3c], name='score4')

    score_res3 = mx.sym.Convolution(data=resnet101_3c, num_filter=150, kernel=(1,1), stride=(1,1), name='score_res3')
    score_fused = score4 + score_res3

    cls_map = mx.sym.slice_axis(data=score_fused, axis=1, begin=0, end=50)
    cls_map = mx.sym.reshape(data=cls_map, name='reshape_cls_map', shape=(0,2,-1,0))

    cls_map = mx.sym.softmax(data=cls_map, name='cls_map_prob', axis=1)

    reg_map = mx.sym.slice_axis(data=score_fused, axis=1, begin=50, end=150)

    net = mx.sym.Group([cls_map, reg_map])

    return net

def copy_params(arg_params, aux_params, exec_):
    for arg_name, arg_param in arg_params.iteritems():
        arg_param.copyto(exec_.arg_dict[arg_name])

    for aux_name, aux_param in aux_params.iteritems():
        aux_param.copyto(exec_.aux_dict[aux_name])

def load_params_checkpoint(prefix, epoch):
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (arg_params, aux_params)

def save_checkpoint(prefix, batch, epoch, symbol, arg_params, aux_params):
    if symbol is not None:
        symbol.save('%s-symbol.json' % prefix)

    save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    param_name = '%s-%05d-%04d.params' % (prefix, batch, epoch)
    mx.nd.save(param_name, save_dict)
    logging.info('Saved checkpoint to \"%s\"', param_name)

def draw_boxes(img, boxes, name):
    boxes = boxes.astype(np.int32)
    for i in range(boxes.shape[0]):
        box = boxes[i,:]
        cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (255,0,0))

    cv2.imwrite('{}.jpg'.format(name), img)

