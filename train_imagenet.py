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
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from data_prefetcher import DataPrefetcher
from model import NetworkImageNet as Network
from torchvision import datasets


parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='data/imagenet/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
# parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr_scheduler')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='ImageNet', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--recover', type=str, default=None, help='load weight and recover previous training')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loader workers')
# bn no decay
parser.add_argument('--bn_no_decay', action='store_true', default=False, help='do not apply weight decay to BN layers')
# fp16, a.k.a half precision
parser.add_argument('--debug', action='store_true', default=False, help='log debug info')
args = parser.parse_args()

args.save = 'checkpoints/eval-{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), args.arch)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if args.debug else logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000

states = {}
if args.recover:
    states = torch.load(args.recover)


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    num_gpus = torch.cuda.device_count()

    genotype = eval("genotypes.%s" % args.arch)
    logging.info('genotype: %s', genotype)
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    if num_gpus > 1:
        model = nn.DataParallel(model)
        model = model.to('cuda')
        logging.info('single GPU training')
        logging.info('parallel training with %d GPUs', num_gpus)
    else:
        model = model.to('cuda')
        logging.info('parallel training with %d GPUs', num_gpus)

    if args.recover:
        model.load_state_dict(states['state_dict'])

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to('cuda')
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.to('cuda')

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

    if args.recover:
        optimizer.load_state_dict(states['optimizer'])

    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = datasets.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    init_epoch = 0
    best_acc_top1 = 0
    best_acc_top5 = 0
    if args.recover:
        init_epoch = states['epoch']
        best_acc_top1 = states['best_acc_top1']

    for epoch in range(init_epoch, args.epochs):
        if args.lr_scheduler == 'cosine':
            logging.info('using cosine lr scheduler')
            scheduler.step(epoch)
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            logging.info('using linear lr scheduler')
            current_lr = adjust_lr(optimizer, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)

        logging.info('epoch %d lr %e', epoch, current_lr)
        logging.debug('%s', optimizer)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr * (epoch + 1) / 5.0
            logging.info('warming-up epoch: %d, lr: %e', epoch, current_lr * (epoch + 1) / 5.0)
        if num_gpus > 1:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        epoch_start = time.time()

        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
        logging.info('train_acc %.4f', train_acc)

        with torch.no_grad():
            valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)

        logging.info('valid_acc_top1 %.4f', valid_acc_top1)
        logging.info('valid_acc_top5 %.4f', valid_acc_top5)

        logging.info('epoch %d overall train_acc=%.4f valid_acc_top1=%.4f valid_acc_top5=%.4f',
                                 epoch, train_acc, valid_acc_top1, valid_acc_top5)

        # gpu info
        utils.gpu_usage(args.debug)

        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds.', epoch_duration)

        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

        logging.info('best performance: top1=%.4f top5=%.4f', best_acc_top1, best_acc_top5)

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save)

def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    start_time = 0
    model.train()

    total_steps = len(train_queue)
    train_queue_iter = DataPrefetcher(train_queue)
    for step in range(total_steps):
        x, y = train_queue_iter.next()
        n = x.size(0)

        b_start = time.time()

        optimizer.zero_grad()
        logits, logits_aux = model(x)
        loss = criterion(logits, y)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, y)
            loss += args.auxiliary_weight*loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('train %05d loss=%.4e top1-acc=%.4f top5-acc=%.4f', step, objs.avg, top1.avg, top5.avg)
            logging.info('duration: %.2fs, average time per batch: %.3fs', duration, batch_time.avg)

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
            logging.info('valid %05d loss=%.4e top1-acc=%.4f top5-acc=%.4f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
