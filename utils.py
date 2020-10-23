import logging

import numpy as np
import os
import pickle
import shutil
import torch

from torch.nn import functional
from torchvision import transforms

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


def binary_acc(output, target):
    with torch.no_grad():
        output = torch.sigmoid(output)
        output_bi = torch.zeros_like(output)
        output_bi[output > 0.5] = 1.0
        total = target.size(0)
        correct = (output_bi * target).sum() + ((1 - output_bi) * (1 - target)).sum()
        acc = correct / total * 100
    return acc


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def rotate_img(image, rotation):
    if rotation == 0: # 0 degrees rotation
        return image
    elif rotation == 1: # 90 degrees rotation
        return transforms.functional.rotate(image, 90)
    elif rotation == 2: # 90 degrees rotation
        return transforms.functional.rotate(image, 180)
    elif rotation == 3: # 270 degrees rotation
        return transforms.functional.rotate(image, 270)
    else:
        raise ValueError('rotation should be 0, 1, 2, or 3')


def data_transforms_cifar10(cutout=False, cutout_length=16, flip=True):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    normalize = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4)])

    if flip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())

    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, valid_transform


# for search only
def data_transforms_imagenet(resize=256, crop=224, padding=32, flip=True, cutout=False, cutout_length=16):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    # train_transform
    train_transform = transforms.Compose([transforms.Resize(resize),
                                          transforms.RandomCrop(crop, padding=padding)])
    if flip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.4,
                                                             contrast=0.4,
                                                             saturation=0.4,
                                                             hue=0.2))
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))

    # valid_transform
    valid_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))

def pickle_save(obj, obj_path):
    with open(obj_path, 'wb') as fp:
        pickle.dump(obj, fp)

def pickle_load(obj_path):
    with open(obj_path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def gumbel(size):
    return -(-torch.rand(size).log()).log().requires_grad_(False)


def gumbel_like(x):
    return -(-torch.rand_like(x).log()).log().requires_grad_(False)


def gumbel_softmax(x, tau=0.1, dim=-1, g=None):
    if g is None:
        g = gumbel_like(x)
    return functional.softmax((x + g) / tau, dim=dim)


def split_bn_params(model):

    def get_bn_params(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            return list(module.parameters())
        else:
            params = []
            for layer in module.children():
                params.extend(get_bn_params(layer))
            return params

    bn_params = get_bn_params(model)
    other_params = [p for p in list(model.parameters()) if p not in set(bn_params)]

    return bn_params, other_params


def gpu_usage(debug=False):
    if debug:
        logging.debug('Device name: %s' % torch.cuda.get_device_name(0))
        logging.debug('Memory usage:')
        logging.debug('  Allocated: %.4f GB' % (torch.cuda.memory_allocated(0) / 1024 ** 3))
        logging.debug('  Cached:    %.4f GB' % (torch.cuda.memory_cached(0) / 1024 ** 3))
    else:
        print('Device name: %s' % torch.cuda.get_device_name(0))
        print('Memory usage:')
        print('  Allocated: %.4f GB' % (torch.cuda.memory_allocated(0) / 1024 ** 3))
        print('  Cached:    %.4f GB' % (torch.cuda.memory_cached(0) / 1024 ** 3))


def log_genotype(model):
    # log genotype (i.e. alpha)
    logging.info('genotype = %s', model.genotype())
    logging.info('alphas_normal: %s\n%s', torch.argmax(model.alphas_normal, dim=-1), model.alphas_normal)
    logging.info('alphas_reduce: %s\n%s', torch.argmax(model.alphas_reduce, dim=-1), model.alphas_reduce)


def layer_dtype(module, level=0):
    logging.debug('%s%s %s', '\t' * level, type(module), [p.dtype for p in module.parameters()] if len(list(module.children())) == 0 else '')
    for layer in module.children():
        layer_dtype(layer, level+1)
