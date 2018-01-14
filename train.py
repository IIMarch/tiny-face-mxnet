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
random.seed(1)


def get_symbol_softmax_loss(is_train=True):
    data = mx.sym.var(name='data')
    labels = mx.sym.var(name='labels')
    bbox_targets = mx.sym.var(name='bbox_targets')
    bbox_inside_weights = mx.sym.var(name='bbox_inside_weights')

    resnet101 = get_symbol_data(data, num_classes=1000, num_layers=101, image_shape=(3,500,500))
    internals = resnet101.get_internals()
    resnet101_22b = internals['_plus29_output']
    resnet101_3c = internals['_plus6_output']

    score_res4 = mx.sym.Convolution(data=resnet101_22b, num_filter=150, kernel=(1,1), stride=(1,1), pad=(0,0), name='score_res4')
    bilinear_weight = mx.sym.Variable(name='bilinear_weight', init=mx.init.Bilinear(), attr={'lr_mult':'0.0'})
    score4_up = mx.sym.UpSampling(*[score_res4, bilinear_weight], num_filter=150, scale=2, sample_type='bilinear', name='score4_up', num_args=2)
    score4 = mx.sym.Crop(*[score4_up, resnet101_3c], name='score4')

    score_res3 = mx.sym.Convolution(data=resnet101_3c, num_filter=150, kernel=(1,1), stride=(1,1), name='score_res3')
    score_fused = score4 + score_res3

    cls_map = mx.sym.slice_axis(data=score_fused, axis=1, begin=0, end=50)
    labels = mx.sym.reshape(data=labels, name='reshape_label', shape=(0,1,25*63,63))
    cls_map = mx.sym.reshape(data=cls_map, name='reshape_cls_map', shape=(0,2,-1,0))
    if not is_train:
        cls_map = mx.sym.softmax(data=cls_map, axis=1, name='cls_map_prob')
    reg_map = mx.sym.slice_axis(data=score_fused, axis=1, begin=50, end=150)

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_map, label=labels, \
                ignore_label=-1, use_ignore=True, grad_scale=1, multi_output=True, \
                normalization='valid', name="cls_prob")

    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
                data=bbox_inside_weights * (reg_map - bbox_targets), scalar=3.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
                normalization='valid', name="loc_loss")

    cls_label = mx.sym.MakeLoss(data=labels, grad_scale=0, name="cls_label")
    cls_pred = mx.sym.MakeLoss(data=cls_map, grad_scale=0, name='cls_pred')

    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, cls_pred])
    #out = mx.symbol.Group([score4_up, labels, bbox_targets, bbox_inside_weights])
    return out

def get_symbol_focal_loss():
    data = mx.sym.var(name='data')
    labels = mx.sym.var(name='labels')
    bbox_targets = mx.sym.var(name='bbox_targets')
    bbox_inside_weights = mx.sym.var(name='bbox_inside_weights')

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
    labels = mx.sym.reshape(data=labels, name='reshape_label', shape=(-1,25*63,63))

    cls_prob = mx.sym.pick(data=cls_map, index=labels, axis=1)
    cls_prob = mx.sym.reshape(data=cls_prob, shape=(-1,))
    focal_loss_ =  100 * mx.sym.mean(- 0.25 * mx.sym.pow(1 - cls_prob, 2) * mx.sym.log(mx.sym.maximum(cls_prob, 1e-10)))
    #focal_loss_ =  1000 * mx.sym.mean(- 0.25 * mx.sym.pow(1 - cls_prob, 10) * mx.sym.log(mx.sym.maximum(cls_prob, 1e-10)))
    focal_loss = mx.sym.MakeLoss(data=focal_loss_, grad_scale=1.0, name='focal_loss')

    reg_map = mx.sym.slice_axis(data=score_fused, axis=1, begin=50, end=150)
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
                data=bbox_inside_weights * (reg_map - bbox_targets), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
                normalization='valid', name="loc_loss")

    cls_label = mx.sym.MakeLoss(data=labels, grad_scale=0, name="cls_label")
    cls_pred = mx.sym.MakeLoss(data=cls_map, grad_scale=0, name='cls_pred')

    net = mx.sym.Group([focal_loss, loc_loss])

    return net


def _update_params(param_arrays, grad_arrays, updater, num_device, kvstore=None, param_names=None):
    for i, pair in enumerate(zip(param_arrays, grad_arrays)):
        arg_list, grad_list = pair
        if grad_list[0] is None:
            continue
        index = i
        if 'bilinear_weight' in param_names[i]:
            continue
        if kvstore:
            pass
        for k, p in enumerate(zip(arg_list, grad_list)):
            w, g = p
            updater(index*num_device+k, g, w)

def get_input_shapes(batch_size):
    N = batch_size
    input_shapes = {'data': (N, 3, 500, 500), 'labels': (N,25,63,63),
            'bbox_targets':(N,100,63,63), 'bbox_inside_weights':(N,100,63,63)}
    return input_shapes

def delete_params_by_shape(net, arg_params, aux_params, input_shapes, initializer):
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(**input_shapes)
    arg_names = net.list_arguments()
    aux_names = net.list_auxiliary_states()
    param_names = [x for x in arg_names if x not in input_shapes]
    param_shapes = []
    for name,shape in zip(arg_names, arg_shapes):
        if name in param_names:
            param_shapes.append(shape)

    for (arg_name, arg_shape) in zip(param_names, param_shapes):
        if arg_name not in arg_params:
            print '{} not in'.format(arg_name)
            arg_params[arg_name] = mx.nd.zeros(arg_shape)
            if 'bilinear_weight' == arg_name:
                initer = mx.init.Bilinear()
                initer(mx.init.InitDesc(arg_name), arg_params[arg_name])
            else:
                initializer(mx.init.InitDesc(arg_name), arg_params[arg_name])
        else:
            if arg_params[arg_name].shape != arg_shape:
                print '{} shape not same'.format(arg_name)
                del arg_params[arg_name]
                arg_params[arg_name] = mx.nd.zeros(arg_shape)
                initializer(mx.init.InitDesc(arg_name), arg_params[arg_name])

    for (aux_name, aux_shape) in zip(aux_names, aux_shapes):
        if aux_name not in aux_params:
            print '{} not in'.format(aux_name)
            aux_params[aux_name] = mx.nd.zeros(aux_shape)
            initializer(mx.init.InitDesc(aux_name), aux_params[aux_name])
        else:
            if aux_params[aux_name].shape != aux_shape:
                print '{} shape not same'.format(aux_name)
                del aux_params[aux_name]
                aux_params[aux_name] = mx.nd.zeros(aux_shape)
                initializer(mx.init.InitDesc(aux_name), aux_params[aux_name])

    for param_arg_names in arg_params.keys():
        if param_arg_names not in arg_names:
            print 'del {}'.format(param_arg_names)
            del arg_params[param_arg_names]

    for param_aux_names in aux_params.keys():
        if param_aux_names not in aux_names:
            print 'del {}'.format(param_aux_names)
            del aux_params[param_aux_names]

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

def load_data(batch, exec_):
    ctx = mx.gpu(2)
    data = []
    labels = []
    bbox_targets = []
    bbox_inside_weights = []
    for i in range(len(batch)):
        data.append(batch[i]['img'])
        labels.append(batch[i]['labels'])
        bbox_targets.append(batch[i]['bbox_targets'])
        bbox_inside_weights.append(batch[i]['bbox_inside_weights'])

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    bbox_targets = np.concatenate(bbox_targets, axis=0)
    bbox_inside_weights = np.concatenate(bbox_inside_weights, axis=0)
    input_data = {'data': mx.nd.array(data, ctx=ctx),
            'labels': mx.nd.array(labels, ctx=ctx),
            'bbox_targets': mx.nd.array(bbox_targets, ctx=ctx),
            'bbox_inside_weights': mx.nd.array(bbox_inside_weights, ctx=ctx)}

    for input_name, input_value in input_data.iteritems():
        input_value.copyto(exec_.arg_dict[input_name])

def save_checkpoint(prefix, batch, epoch, symbol, arg_params, aux_params):
    if symbol is not None:
        symbol.save('%s-symbol.json' % prefix)

    save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    param_name = '%s-%05d-%04d.params' % (prefix, batch, epoch)
    mx.nd.save(param_name, save_dict)
    logging.info('Saved checkpoint to \"%s\"', param_name)

def run(mxIter):
    model_prefix = '/data2/obj_detect/imagenet_models/resnet/resnet-101'
    load_epoch = 0
    #model_prefix = './stage1_models/tiny_face-06440'
    #load_epoch = 42
    #model_prefix = './tiny_face-06440'
    #load_epoch = 140
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    input_shapes = get_input_shapes(mxIter.batch_size)
    optimizer = 'sgd'
    optimizer_params = {
        'learning_rate': 0.0001,
        'momentum' : 0.90,
        'wd' : 0.0001}
    optimizer = opt.create(optimizer, rescale_grad=1.0 / mxIter.batch_size, **optimizer_params)
    updater = get_updater(optimizer)

    net = get_symbol_focal_loss()
    arg_params, aux_params = load_params_checkpoint(model_prefix, load_epoch)
    arg_names = net.list_arguments()
    param_names = [x for x in arg_names if x not in input_shapes]

    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
    delete_params_by_shape(net, arg_params, aux_params, input_shapes, initializer)
    exec_ = net.simple_bind(ctx=mx.gpu(2), **input_shapes)
    copy_params(arg_params, aux_params, exec_)

    param_arrays = [[exec_.arg_arrays[i]] for i,name in enumerate(arg_names) if name in param_names]
    grad_arrays = [[exec_.grad_arrays[i]] for i,name in enumerate(arg_names) if name in param_names]

    #monitor = mx.monitor.Monitor(interval=1, pattern='.*backward.*')
    #monitor.install(exec_)

    batch_size = mxIter.batch_size
    for epoch in range(load_epoch+1, 200):
        num_batch = 0
        metric = 0
        num_inst = 0
        num_reg_inst = 0
        reg_metric = 0
        for batch in mxIter:
            load_data(batch, exec_)
            #monitor.tic()
            exec_.forward(is_train=True)
            outputs = [output.asnumpy() for output in exec_._get_outputs()]
            exec_.backward()
            #monitor.toc_print()
            _update_params(param_arrays, grad_arrays, updater, 1, param_names=param_names)
            num_batch += 1

            # metric
            metric += np.sum(outputs[0])
            reg_metric += np.sum(outputs[1])
            print 'batch -> {}'.format(num_batch)
            print 'focal_loss -> {}'.format(metric / num_batch)
            print 'l1_loss -> {}'.format(reg_metric / num_batch)

            if num_batch % 1000 == 0:
                save_arg_params = {}
                for param_name in param_names:
                    save_arg_params[param_name] = exec_.arg_dict[param_name]
                save_aux_params = exec_.aux_dict
                save_checkpoint('./tiny_face', num_batch, epoch, net, save_arg_params, save_aux_params)

        mxIter.reset()
        save_arg_params = {}
        for param_name in param_names:
            save_arg_params[param_name] = exec_.arg_dict[param_name]
        save_aux_params = exec_.aux_dict
        save_checkpoint('./tiny_face', num_batch, epoch, net, save_arg_params, save_aux_params)


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4

    return bbox_transform(ex_rois, gt_rois).astype(np.float32, copy=False)

class widerface(object):

    def __init__(self, isTrain):
        if isTrain:
            self.path = './wider_face_train.lst'
            self.im_base_path = './data/WIDER_FACE/WIDER_train/images'
        else:
            self.path = './wider_face_val.lst'

    def read(self):
        if os.path.exists('./wider_face_train.pkl'):
            fid = open('./wider_face_train.pkl', 'rb')
            self.roidb = cPickle.load(fid)

        roidb = []
        with open(self.path, 'r') as fid:
            for line in fid:
                splited = line.split(' ')
                if len(splited) == 1:
                    continue
                img_path = os.path.join(self.im_base_path, splited[0]+'.jpg')
                xywh = np.array([int(i) for i in splited[1:]])
                xywh = xywh.reshape(-1,4)
                bboxes = np.zeros_like(xywh)
                bboxes[:,0] = xywh[:,0]
                bboxes[:,1] = xywh[:,1]
                bboxes[:,2] = xywh[:,0] + xywh[:,2] - 1
                bboxes[:,3] = xywh[:,1] + xywh[:,3] - 1
                ret = {'img_path':img_path, 'bboxes':bboxes.astype(np.float)}
                roidb.append(ret)

        return roidb

def draw_boxes(img, boxes, name):
    boxes = boxes.astype(np.int32)
    for i in range(boxes.shape[0]):
        box = boxes[i,:]
        cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (255,0,0))

    cv2.imwrite('{}.jpg'.format(name), img)

class RoidbIter:
    def __init__(self, roidb, batch_size=2, shuffle=True):
        self.num_inst = len(roidb)
        self.perm = range(self.num_inst)
        self.batch_size = batch_size
        self.cur = 0
        self.shuffle = shuffle
        self.roidb = roidb
        self.all_anchors = get_anchors()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            random.shuffle(self.perm)

    def __iter__(self):
        return self

    def next(self):
        if self.cur + self.batch_size > self.num_inst:
            raise StopIteration
        else:
            idx = self.perm[self.cur:self.cur+self.batch_size]
            self.cur = self.cur + self.batch_size
            batch = [self.roidb[r] for r in idx]
            data_batch = self._proc(batch)
            return data_batch

    def _proc(self, batch):
        input_size = 500
        A = 25
        height = 63
        width = 63
        all_anchors = self.all_anchors

        data_batch = []
        for i in range(len(batch)):
            im = cv2.imread(batch[i]['img_path'])
            assert(im is not None)
            assert(len(im.shape) == 3)
            #draw_boxes(im.copy(), batch[i]['bboxes'], 'ori')

            bboxes = batch[i]['bboxes'].copy()
            imageBuffer = np.zeros([input_size, input_size, 3])
            rnd = random.random()
            if rnd < 1.0 / 3:
                bboxes /= 2.0;
                im = cv2.resize(im, (0,0), fx=0.5, fy=0.5)
            elif rnd > 2.0 / 3:
                bboxes *= 2.0;
                im = cv2.resize(im, (0,0), fx=2, fy=2)
            else:
                pass

            #crop image
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

            try:
                imageBuffer[paste_y1:paste_y2+1, paste_x1:paste_x2+1, :] = im[crop_y1:crop_y2+1, crop_x1:crop_x2+1, :]
            except:
                import ipdb; ipdb.set_trace()

            assert(bboxes.shape[0] != 0)

            #draw_boxes(im.copy(), bboxes, 'resize')
            ori_bboxes = bboxes.copy()
            bboxes[:,0] = np.maximum(bboxes[:,0], crop_x1)
            bboxes[:,1] = np.maximum(bboxes[:,1], crop_y1)
            bboxes[:,2] = np.minimum(bboxes[:,2], crop_x2)
            bboxes[:,3] = np.minimum(bboxes[:,3], crop_y2)

            #draw_boxes(im.copy(), bboxes, 'crop')

            tovlp = bbox_overlaps(bboxes, ori_bboxes)
            argmax_tovlp = tovlp.argmax(axis=1)
            max_toplp = tovlp[np.arange(tovlp.shape[0]), argmax_tovlp]

            labelRect = ori_bboxes.copy()
            labelRect[:,0] -= (crop_x1 - paste_x1)
            labelRect[:,1] -= (crop_y1 - paste_y1)
            labelRect[:,2] -= (crop_x1 - paste_x1)
            labelRect[:,3] -= (crop_y1 - paste_y1)

            labelRect[:,0] = np.minimum(input_size, np.maximum(0, labelRect[:,0]))
            labelRect[:,1] = np.minimum(input_size, np.maximum(0, labelRect[:,1]))
            labelRect[:,2] = np.minimum(input_size, np.maximum(0, labelRect[:,2]))
            labelRect[:,3] = np.minimum(input_size, np.maximum(0, labelRect[:,3]))

            #draw_boxes(imageBuffer.copy(), labelRect, 'move')

            invalid_idx = np.logical_or(labelRect[:,2] <= labelRect[:,0], labelRect[:,3] <= labelRect[:,1])
            invalid_idx = np.logical_or(invalid_idx, max_toplp < 0.3)
            invalid_idx = np.where(invalid_idx == True)

            gt_boxes = np.delete(labelRect, invalid_idx[0], axis=0)
            blob = imageBuffer[:,:,::-1].transpose(2,0,1)

            # focal loss can use all 0
            labels = np.empty((all_anchors.shape[0],), dtype=np.float32)
            labels.fill(0)

            #draw_boxes(imageBuffer.copy(), gt_boxes, '{}_{}'.format(self.cur, i))

            if gt_boxes.shape[0] > 0:
                overlaps = bbox_overlaps(np.ascontiguousarray(all_anchors, dtype=np.float),
                        np.ascontiguousarray(gt_boxes, dtype=np.float))
                argmax_overlaps = overlaps.argmax(axis=1) # (ex,)
                max_overlaps = overlaps[np.arange(all_anchors.shape[0]), argmax_overlaps]
                gt_argmax_overlaps = overlaps.argmax(axis=0)
                gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]

                # fg and bg selecting strategy
		neg_idx = np.logical_and(0 <= max_overlaps, max_overlaps < 0.5)
		labels[neg_idx] = 0

                gt_fg_idx = gt_argmax_overlaps[gt_max_overlaps > 0.2]

                labels[gt_fg_idx] = 1
                labels[max_overlaps >= 0.5] = 1

                #neg_idx = np.logical_and(0 <= max_overlaps, max_overlaps < 0.5)
                #labels[neg_idx] = 0

            else:
                labels[:] = 0

            ''' comment for softmax loss
            num_fg = int(256 * 0.5)
            fg_inds = np.where(labels == 1)[0]

            if len(fg_inds) > num_fg:
                disable_inds = npr.choice(
                    fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                labels[disable_inds] = -1

            num_bg = 256 - np.sum(labels == 1)
            bg_inds = np.where(labels == 0)[0]

            if len(bg_inds) > num_bg:
                disable_inds = npr.choice(
                    bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                labels[disable_inds] = -1
            '''

            bbox_targets = np.zeros((all_anchors.shape[0], 4), dtype=np.float32)

            if gt_boxes.size > 0:
                bbox_targets = _compute_targets(all_anchors, gt_boxes[argmax_overlaps, :])

            #print '------------bbox_targets-------------'
            #print bbox_targets[np.where(labels==1)[0],:]
            bbox_inside_weights = np.zeros((all_anchors.shape[0], 4), dtype=np.float32)
            bbox_inside_weights[labels == 1, :] = np.array([1.0, 1.0, 1.0, 1.0])

            labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
            bbox_targets = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
            bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

            data_batch.append({'labels':labels,
                'img': blob[np.newaxis, :, :, :],
                'bbox_targets': bbox_targets,
                'bbox_inside_weights': bbox_inside_weights})

        return data_batch

if __name__ == '__main__':
    wof = widerface(True)
    roidb = wof.read()
    roidbIter = RoidbIter(roidb, batch_size=2)
    run(roidbIter)
    #for batch in roidbIter:
    #    pass




