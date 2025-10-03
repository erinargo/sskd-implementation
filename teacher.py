import os
import os.path as osp
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from tensorboardX import SummaryWriter

from utils import AverageMeter, accuracy   # helper functions for tracking metrics
from models import model_dict              # dictionary of models (defined elsewhere)

# Enable cudnn autotuner to select the fastest convolution algorithm
torch.backends.cudnn.benchmark = True


def main():
    # ---------------------------
    # Parse training arguments
    # ---------------------------
    parser = argparse.ArgumentParser(description='Train teacher network on CIFAR100.')
    parser.add_argument('--epoch', type=int, default=240)             # total number of epochs
    parser.add_argument('--batch-size', type=int, default=64)         # training batch size
    parser.add_argument('--lr', type=float, default=0.05)             # learning rate
    parser.add_argument('--momentum', type=float, default=0.9)        # SGD momentum
    parser.add_argument('--weight-decay', type=float, default=5e-4)   # L2 regularization
    parser.add_argument('--gamma', type=float, default=0.1)           # LR decay factor
    parser.add_argument('--milestones', type=int, nargs='+', default=[150,180,210])  # LR schedule

    parser.add_argument('--save-interval', type=int, default=40)      # save checkpoint frequency
    parser.add_argument('--arch', type=str)                           # model architecture key
    parser.add_argument('--seed', type=int, default=0)                # random seed for reproducibility
    parser.add_argument('--gpu-id', type=int, default=0)              # GPU device id

    args = parser.parse_args()

    # ---------------------------
    # Set random seeds
    # ---------------------------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Choose which GPU to use
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Experiment setup (directory to save logs and checkpoints)
    exp_name = 'teacher_{}_seed{}'.format(args.arch, args.seed)
    exp_path = './experiments/{}'.format(exp_name)
    os.makedirs(exp_path, exist_ok=True)

    # ---------------------------
    # Data preprocessing
    # ---------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),               # data augmentation
        transforms.RandomHorizontalFlip(),                  # random mirror
        transforms.ToTensor(),                              # convert to tensor
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], # normalize with CIFAR100 stats
                             std=[0.2675, 0.2565, 0.2761]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                             std=[0.2675, 0.2565, 0.2761]),
    ])

    # Load CIFAR100 train/test datasets
    trainset = CIFAR100('./data', train=True, transform=transform_train, download=True)
    valset = CIFAR100('./data', train=False, transform=transform_test, download=True)

    # Create DataLoaders
    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(valset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=False)

    # ---------------------------
    # Model, optimizer, scheduler
    # ---------------------------
    model = model_dict[args.arch](num_classes=100).cuda()   # load chosen architecture
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    # TensorBoard logger
    logger = SummaryWriter(osp.join(exp_path, 'events'))
    best_acc = -1   # track best validation accuracy

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(args.epoch):
        model.train()
        loss_record = AverageMeter()  # track average training loss
        acc_record = AverageMeter()   # track average training accuracy

        start = time.time()
        for x, target in train_loader:
            optimizer.zero_grad()

            # Move batch to GPU
            x = x.cuda()
            target = target.cuda()

            # Forward pass
            output = model(x)
            loss = F.cross_entropy(output, target)  # cross entropy loss

            # Backward pass and parameter update
            loss.backward()
            optimizer.step()

            # Compute training accuracy for this batch
            batch_acc = accuracy(output, target, topk=(1,))[0]

            # Update running averages
            loss_record.update(loss.item(), x.size(0))
            acc_record.update(batch_acc.item(), x.size(0))

        # Log training metrics to TensorBoard
        logger.add_scalar('train/cls_loss', loss_record.avg, epoch+1)
        logger.add_scalar('train/cls_acc', acc_record.avg, epoch+1)

        run_time = time.time() - start

        # Print training progress
        info = 'train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t cls_loss:{:.3f}\t cls_acc:{:.2f}\t'.format(
            epoch+1, args.epoch, run_time, loss_record.avg, acc_record.avg)
        print(info)

        # ---------------------------
        # Validation loop
        # ---------------------------
        model.eval()
        acc_record = AverageMeter()
        loss_record = AverageMeter()
        start = time.time()

        for x, target in val_loader:
            x = x.cuda()
            target = target.cuda()
            with torch.no_grad():
                output = model(x)
                loss = F.cross_entropy(output, target)

            # Compute validation accuracy
            batch_acc = accuracy(output, target, topk=(1,))[0]
            loss_record.update(loss.item(), x.size(0))
            acc_record.update(batch_acc.item(), x.size(0))

        run_time = time.time() - start

        # Log validation metrics
        logger.add_scalar('val/cls_loss', loss_record.avg, epoch+1)
        logger.add_scalar('val/cls_acc', acc_record.avg, epoch+1)

        # Print validation progress
        info = 'test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t cls_loss:{:.3f}\t cls_acc:{:.2f}\n'.format(
                epoch+1, args.epoch, run_time, loss_record.avg, acc_record.avg)
        print(info)

        # Update learning rate scheduler
        scheduler.step()

        # ---------------------------
        # Save model checkpoints
        # ---------------------------
        # Periodic checkpoint saving
        if (epoch+1) in args.milestones or epoch+1 == args.epoch or (epoch+1) % args.save_interval == 0:
            state_dict = dict(epoch=epoch+1, state_dict=model.state_dict(), acc=acc_record.avg)
            name = osp.join(exp_path, 'ckpt/{:03d}.pth'.format(epoch+1))
            os.makedirs(osp.dirname(name), exist_ok=True)
            torch.save(state_dict, name)

        # Save best-performing model
        if acc_record.avg > best_acc:
            state_dict = dict(epoch=epoch+1, state_dict=model.state_dict(), acc=acc_record.avg)
            name = osp.join(exp_path, 'ckpt/best.pth')
            os.makedirs(osp.dirname(name), exist_ok=True)
            torch.save(state_dict, name)
            best_acc = acc_record.avg

        print('best_acc: {:.2f}'.format(best_acc))


# ---------------------------
# Script entry point
# ---------------------------

## MILEAGE MAY VARY IF RUNNING ON MAC
if __name__ == "__main__":
    import torch
    from torch.multiprocessing import set_start_method

    # For DataLoader multiprocessing compatibility on some systems
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    main()
