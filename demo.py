import find_mxnet
import os, sys, random
import cv2
from bbox_transform import bbox_transform_inv
import logging
from test import get_symbol, copy_params, load_params_checkpoint, draw_boxes
import mxnet as mx
import random
import numpy as np
from anchor import get_anchors
from nms.nms_wrapper import nms
class Predictor(object):
    def __init__(self):
        model_prefix = './tiny_face-06440'
        load_epoch = 88
        arg_params, aux_params = load_params_checkpoint(model_prefix, load_epoch)
        net = get_symbol()
        cls_map = net.get_internals()['cls_map_prob_output']
        reg_map = net.get_internals()['slice_axis1_output']
        net = mx.sym.Group([cls_map, reg_map])
        input_shapes = {'data': (1,3,500,500)}
        self.exec_ = net.simple_bind(ctx=mx.gpu(2), **input_shapes)
        copy_params(arg_params, aux_params, self.exec_)
        self.anchors = get_anchors()

    def forward(self, img_path, i):
        im = cv2.imread(img_path)
        input_size = 500
        imageBuffer = np.zeros([input_size, input_size, 3])

        crop_y1 = random.randint(0, max(0, im.shape[0]-input_size))
        crop_x1 = random.randint(0, max(0, im.shape[1]-input_size))
        crop_y2 = min(im.shape[0]-1, crop_y1+input_size-1)
        crop_x2 = min(im.shape[1]-1, crop_x1+input_size-1)

        crop_h = crop_y2-crop_y1+1
        crop_w = crop_x2-crop_x1+1

        paste_y1 = random.randint(0, input_size-crop_h)
        paste_x1 = random.randint(0, input_size-crop_w)
        paste_y2 = paste_y1 + crop_h - 1
        paste_x2 = paste_x1 + crop_w - 1

        imageBuffer[paste_y1:paste_y2+1, paste_x1:paste_x2+1, :] = im[crop_y1:crop_y2+1, crop_x1:crop_x2+1, :]

        cv2.imwrite('input.jpg', imageBuffer)

        blob = imageBuffer[:,:,::-1].transpose(2,0,1)
        blob = mx.nd.array(blob[np.newaxis, :, :, :])
        blob.copyto(self.exec_.arg_dict['data'])

        self.exec_.forward(is_train=False)

        outputs = [output.asnumpy() for output in self.exec_._get_outputs()]
        cls_map = outputs[0]
        reg_map = outputs[1]
        bbox_deltas = reg_map.transpose((0, 2, 3, 1)).reshape((-1,4))
        scores = cls_map[0,1:2,:,:].reshape((1,25,63,63)) # (1,1,1575,63)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        proposals = bbox_transform_inv(self.anchors, bbox_deltas)
        #proposals = self.anchors
        #draw_boxes(imageBuffer, proposals[:100], 'res1')
        order = scores.ravel().argsort()[::-1]
        order = order[:6000]
        scores = scores[order]
        proposals = proposals[order, :]
        keep = nms(np.hstack((proposals, scores)), 0.05)

        keep = keep[:300]
        proposals = proposals[keep, :]
        scores = scores[keep]

        keep = np.where(scores > 0.4)[0]
        proposals = proposals[keep, :]
        scores = scores[keep]

        draw_boxes(imageBuffer, proposals, 'res_{}'.format(i))


if __name__ == '__main__':
    predictor = Predictor()
    for i in range(10):
        random.seed(i)
        predictor.forward('/data2/obj_detect/tiny/data/demo/selfie.jpg', i)
