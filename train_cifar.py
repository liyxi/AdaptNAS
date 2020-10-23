import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.backends.cudnn as cudnn

from data_prefetcher import DataPrefetcher
from model import NetworkCIFAR as Network
from torchvision import datasets


parser = argparse.ArgumentParser("cifar")
# data
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loader workers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# save
parser.add_argument('--position', type=str, default='', help='specify a position, default is current directory')
parser.add_argument('--save', type=str, default='CIFAR', help='experiment name')
parser.add_argument('--recover', type=str, default=None, help='recover path')
# parser.add_argument('--init_epoch', type=int, default=0, help='initial epoch number')
# training setting
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# model setting
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
# bn no decay
parser.add_argument('--bn_no_decay', action='store_true', default=False, help='do not apply weight decay to BN layers')
# others
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--debug', action='store_true', default=False, help='log debug info')
args = parser.parse_args()


args.save = os.path.join(args.position, 'checkpoints/eval-{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), args.arch))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if args.debug else logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.to('cuda')

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to('cuda')

    # not apply weight decay to BN layers
    if args.bn_no_decay:
        logging.info('BN layers are excluded from weight decay')
        bn_params, other_params = utils.split_bn_params(model)
        logging.debug('bn: %s', [p.dtype for p in bn_params])
        logging.debug('other: %s', [p.dtype for p in other_params])
        param_group = [{'params': bn_params, 'weight_decay': 0},
                       {'params': other_params}]
    else:
        param_group = model.parameters()

    optimizer = torch.optim.SGD(
        param_group,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    logging.info('optimizer: %s', optimizer)

    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    train_data = datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    init_epoch = 0
    best_acc = 0

    if args.recover:
        states = torch.load(args.recover)
        model.load_state_dict(states['stete'])
        init_epoch = states['epoch'] + 1
        best_acc = states['best_acc']
        logging.info('checkpoint loaded')
        scheduler.step(init_epoch)
        logging.info('scheduler is set to epoch %d. learning rate is %s', init_epoch, scheduler.get_lr())

    for epoch in range(init_epoch, args.epochs):
        logging.info('epoch %d lr %s', epoch, scheduler.get_lr())
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %.4f', train_acc)

        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

        logging.info('epoch %03d overall train_acc=%.4f valid_acc=%.4f', epoch, train_acc, valid_acc)

        scheduler.step()

        # gpu info
        utils.gpu_usage(args.debug)

        if valid_acc > best_acc:
            best_acc = valid_acc

        logging.info('best acc: %.4f', best_acc)

        utils.save_checkpoint(state={'stete': model.state_dict(),
                                     'epoch': epoch,
                                     'best_acc': best_acc,},
                              is_best=False, save=args.save)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    total_steps = len(train_queue)
    train_queue_iter = DataPrefetcher(train_queue)
    for step in range(total_steps):
        x, y = train_queue_iter.next()
        n = x.size(0)

        optimizer.zero_grad()
        logits, logits_aux = model(x)
        loss = criterion(logits, y)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, y)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %04d loss=%.4e top1-acc=%.4f top5-acc=%.4f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    total_steps = len(valid_queue)
    valid_queue_iter = DataPrefetcher(valid_queue)
    for step in range(total_steps):
        x, y = valid_queue_iter.next()
        n = x.size(0)

        logits, _ = model(x)
        loss = criterion(logits, y)

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %04d loss=%.4e top1-acc=%.4f top5-acc=%.4f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
