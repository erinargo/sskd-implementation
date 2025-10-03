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
from tensorboardX import SummaryWriter

from utils import AverageMeter, accuracy
from wrapper import wrapper
from cifar import CIFAR100

from models import model_dict


def main():
    ## Enable cuDNN benchmark mode for potential performance improvement on fixed input size
    torch.backends.cudnn.benchmark = True

    ## Argument parser: allows configuring training hyperparameters from command line
    parser = argparse.ArgumentParser(description='train SSKD student network.')
    parser.add_argument('--epoch', type=int, default=240)  # total student training epochs
    parser.add_argument('--t-epoch', type=int, default=60)  # teacher self-supervised training epochs
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--lr', type=float, default=0.05)  # student learning rate
    parser.add_argument('--t-lr', type=float, default=0.05)  # teacher learning rate
    parser.add_argument('--momentum', type=float, default=0.9)  # SGD momentum
    parser.add_argument('--weight-decay', type=float, default=5e-4)  # L2 regularization
    parser.add_argument('--gamma', type=float, default=0.1)  # LR decay factor
    parser.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])  # student LR decay schedule
    parser.add_argument('--t-milestones', type=int, nargs='+', default=[30, 45])  # teacher LR decay schedule

    parser.add_argument('--save-interval', type=int, default=40)  # checkpoint save interval
    parser.add_argument('--ce-weight', type=float, default=0.1)  # weight for CE loss
    parser.add_argument('--kd-weight', type=float, default=0.9)  # weight for KD loss
    parser.add_argument('--tf-weight', type=float, default=2.7)  # weight for transformation loss
    parser.add_argument('--ss-weight', type=float, default=10.0)  # weight for self-supervision loss

    parser.add_argument('--kd-T', type=float, default=4.0)  # temperature for KD loss
    parser.add_argument('--tf-T', type=float, default=4.0)  # temperature for LT (transformation) loss
    parser.add_argument('--ss-T', type=float, default=0.5)  # temperature for SS loss

    parser.add_argument('--ratio-tf', type=float, default=1.0)  # keep ratio for wrong TF predictions
    parser.add_argument('--ratio-ss', type=float, default=0.75)  # keep ratio for wrong SS predictions
    parser.add_argument('--s-arch', type=str)  # student architecture name
    parser.add_argument('--t-path', type=str)  # teacher checkpoint path

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu-id', type=int, default=0)

    ## Parse args and set seeds for reproducibility
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    ## Build experiment name string for logging
    t_name = osp.basename(osp.abspath(args.t_path))
    t_arch = '_'.join(t_name.split('_')[1:-1])
    exp_name = 'sskd_student_{}_weight{}+{}+{}+{}_T{}+{}+{}_ratio{}+{}_seed{}_{}'.format( \
        args.s_arch, \
        args.ce_weight, args.kd_weight, args.tf_weight, args.ss_weight, \
        args.kd_T, args.tf_T, args.ss_T, \
        args.ratio_tf, args.ratio_ss, \
        args.seed, t_name)
    exp_path = './experiments/{}'.format(exp_name)
    os.makedirs(exp_path, exist_ok=True)

    ## Data augmentation / normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
    ])

    ## Load CIFAR100 dataset
    trainset = CIFAR100('./data', train=True, transform=transform_train)
    valset = CIFAR100('./data', train=False, transform=transform_test)

    ## Create dataloaders
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

    ## Load teacher model from checkpoint
    ckpt_path = osp.join(args.t_path, 'ckpt/best.pth')
    t_model = model_dict[t_arch](num_classes=100).cuda()
    state_dict = torch.load(ckpt_path)['state_dict']
    t_model.load_state_dict(state_dict)
    t_model = wrapper(module=t_model).cuda()

    ## Teacher optimizer and scheduler (train only projection head for self-supervised pretraining)
    t_optimizer = optim.SGD([{'params': t_model.backbone.parameters(), 'lr': 0.0},
                             {'params': t_model.proj_head.parameters(), 'lr': args.t_lr}],
                            momentum=args.momentum, weight_decay=args.weight_decay)
    t_model.eval()
    t_scheduler = MultiStepLR(t_optimizer, milestones=args.t_milestones, gamma=args.gamma)

    ## Logger (TensorBoard)
    logger = SummaryWriter(osp.join(exp_path, 'events'))

    ## Evaluate teacher initial accuracy before training SSP
    acc_record = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    for x, target in val_loader:
        x = x[:, 0, :, :, :].cuda()  # only first augmentation
        target = target.cuda()
        with torch.no_grad():
            output, _, feat = t_model(x)
            loss = F.cross_entropy(output, target)

        batch_acc = accuracy(output, target, topk=(1,))[0]
        acc_record.update(batch_acc.item(), x.size(0))
        loss_record.update(loss.item(), x.size(0))

    run_time = time.time() - start
    info = 'teacher cls_acc:{:.2f}\n'.format(acc_record.avg)
    print(info)

    # ============================
    # Teacher SSP pretraining loop
    # ============================
    for epoch in range(args.t_epoch):

        t_model.eval()
        loss_record = AverageMeter()
        acc_record = AverageMeter()

        start = time.time()
        for x, _ in train_loader:
            t_optimizer.zero_grad()

            x = x.cuda()
            c, h, w = x.size()[-3:]
            x = x.view(-1, c, h, w)  # flatten views

            ## Forward pass teacher with SSP head
            _, rep, feat = t_model(x, bb_grad=False)
            batch = int(x.size(0) / 4)
            nor_index = (torch.arange(4 * batch) % 4 == 0).cuda()
            aug_index = (torch.arange(4 * batch) % 4 != 0).cuda()

            ## Separate normal and augmented reps
            nor_rep = rep[nor_index]
            aug_rep = rep[aug_index]
            nor_rep = nor_rep.unsqueeze(2).expand(-1, -1, 3 * batch).transpose(0, 2)
            aug_rep = aug_rep.unsqueeze(2).expand(-1, -1, 1 * batch)

            ## Self-supervised similarity classification (CE loss)
            simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
            target = torch.arange(batch).unsqueeze(1).expand(-1, 3).contiguous().view(-1).long().cuda()
            loss = F.cross_entropy(simi, target)

            ## Backprop and optimize SSP head
            loss.backward()
            t_optimizer.step()

            batch_acc = accuracy(simi, target, topk=(1,))[0]
            loss_record.update(loss.item(), 3 * batch)
            acc_record.update(batch_acc.item(), 3 * batch)

        ## Log SSP training stats
        logger.add_scalar('train/teacher_ssp_loss', loss_record.avg, epoch + 1)
        logger.add_scalar('train/teacher_ssp_acc', acc_record.avg, epoch + 1)

        run_time = time.time() - start
        info = 'teacher_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\t'.format(
            epoch + 1, args.t_epoch, run_time, loss_record.avg, acc_record.avg)
        print(info)

        ## Validate SSP head
        t_model.eval()
        acc_record = AverageMeter()
        loss_record = AverageMeter()
        start = time.time()
        for x, _ in val_loader:
            x = x.cuda()
            c, h, w = x.size()[-3:]
            x = x.view(-1, c, h, w)

            with torch.no_grad():
                _, rep, feat = t_model(x)
            batch = int(x.size(0) / 4)
            nor_index = (torch.arange(4 * batch) % 4 == 0).cuda()
            aug_index = (torch.arange(4 * batch) % 4 != 0).cuda()

            nor_rep = rep[nor_index]
            aug_rep = rep[aug_index]
            nor_rep = nor_rep.unsqueeze(2).expand(-1, -1, 3 * batch).transpose(0, 2)
            aug_rep = aug_rep.unsqueeze(2).expand(-1, -1, 1 * batch)
            simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
            target = torch.arange(batch).unsqueeze(1).expand(-1, 3).contiguous().view(-1).long().cuda()
            loss = F.cross_entropy(simi, target)

            batch_acc = accuracy(simi, target, topk=(1,))[0]
            acc_record.update(batch_acc.item(), 3 * batch)
            loss_record.update(loss.item(), 3 * batch)

        run_time = time.time() - start
        logger.add_scalar('val/teacher_ssp_loss', loss_record.avg, epoch + 1)
        logger.add_scalar('val/teacher_ssp_acc', acc_record.avg, epoch + 1)

        info = 'ssp_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\n'.format(
            epoch + 1, args.t_epoch, run_time, loss_record.avg, acc_record.avg)
        print(info)

        t_scheduler.step()

    ## Save teacher model after SSP training
    name = osp.join(exp_path, 'ckpt/teacher.pth')
    os.makedirs(osp.dirname(name), exist_ok=True)
    torch.save(t_model.state_dict(), name)

    # ==================================
    # Student training with SSKD losses
    # ==================================
    s_model = model_dict[args.s_arch](num_classes=100)
    s_model = wrapper(module=s_model).cuda()
    optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    ## Track best student performance
    best_acc = 0
    for epoch in range(args.epoch):

        # =====================
        # Train student network
        # =====================
        s_model.train()
        loss1_record = AverageMeter()  # CE loss
        loss2_record = AverageMeter()  # KD loss
        loss3_record = AverageMeter()  # TF loss
        loss4_record = AverageMeter()  # SS loss
        cls_acc_record = AverageMeter()
        ssp_acc_record = AverageMeter()

        start = time.time()
        for x, target in train_loader:
            optimizer.zero_grad()

            c, h, w = x.size()[-3:]
            x = x.view(-1, c, h, w).cuda()
            target = target.cuda()

            batch = int(x.size(0) / 4)
            nor_index = (torch.arange(4 * batch) % 4 == 0).cuda()
            aug_index = (torch.arange(4 * batch) % 4 != 0).cuda()

            ## Forward pass student
            output, s_feat, _ = s_model(x, bb_grad=True)
            log_nor_output = F.log_softmax(output[nor_index] / args.kd_T, dim=1)
            log_aug_output = F.log_softmax(output[aug_index] / args.tf_T, dim=1)
            with torch.no_grad():
                knowledge, t_feat, _ = t_model(x)
                nor_knowledge = F.softmax(knowledge[nor_index] / args.kd_T, dim=1)
                aug_knowledge = F.softmax(knowledge[aug_index] / args.tf_T, dim=1)

            # ========================
            # Distillation (TF + SS)
            # ========================
            ## Error-level ranking: filter which augmented samples to keep for TF loss
            aug_target = target.unsqueeze(1).expand(-1, 3).contiguous().view(-1).long().cuda()
            rank = torch.argsort(aug_knowledge, dim=1, descending=True)
            rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)
            index = torch.argsort(rank)
            tmp = torch.nonzero(rank, as_tuple=True)[0]
            wrong_num = tmp.numel()
            correct_num = 3 * batch - wrong_num
            wrong_keep = int(wrong_num * args.ratio_tf)
            index = index[:correct_num + wrong_keep]
            distill_index_tf = torch.sort(index)[0]

            ## Self-supervised features (student vs teacher)
            s_nor_feat = s_feat[nor_index]
            s_aug_feat = s_feat[aug_index]
            s_nor_feat = s_nor_feat.unsqueeze(2).expand(-1, -1, 3 * batch).transpose(0, 2)
            s_aug_feat = s_aug_feat.unsqueeze(2).expand(-1, -1, 1 * batch)
            s_simi = F.cosine_similarity(s_aug_feat, s_nor_feat, dim=1)

            t_nor_feat = t_feat[nor_index]
            t_aug_feat = t_feat[aug_index]
            t_nor_feat = t_nor_feat.unsqueeze(2).expand(-1, -1, 3 * batch).transpose(0, 2)
            t_aug_feat = t_aug_feat.unsqueeze(2).expand(-1, -1, 1 * batch)
            t_simi = F.cosine_similarity(t_aug_feat, t_nor_feat, dim=1)

            t_simi = t_simi.detach()
            aug_target = torch.arange(batch).unsqueeze(1).expand(-1, 3).contiguous().view(-1).long().cuda()
            rank = torch.argsort(t_simi, dim=1, descending=True)
            rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)
            index = torch.argsort(rank)
            tmp = torch.nonzero(rank, as_tuple=True)[0]
            wrong_num = tmp.numel()
            correct_num = 3 * batch - wrong_num
            wrong_keep = int(wrong_num * args.ratio_ss)
            index = index[:correct_num + wrong_keep]
            distill_index_ss = torch.sort(index)[0]

            ## Apply temperature softmax for SS loss
            log_simi = F.log_softmax(s_simi / args.ss_T, dim=1)
            simi_knowledge = F.softmax(t_simi / args.ss_T, dim=1)

            # =====================
            # Compute all 4 losses
            # =====================
            loss1 = F.cross_entropy(output[nor_index], target)  # CE
            loss2 = F.kl_div(log_nor_output, nor_knowledge, reduction='batchmean') * args.kd_T * args.kd_T
            loss3 = F.kl_div(log_aug_output[distill_index_tf], aug_knowledge[distill_index_tf], \
                             reduction='batchmean') * args.tf_T * args.tf_T
            loss4 = F.kl_div(log_simi[distill_index_ss], simi_knowledge[distill_index_ss], \
                             reduction='batchmean') * args.ss_T * args.ss_T

            ## Weighted sum
            loss = args.ce_weight * loss1 + args.kd_weight * loss2 + args.tf_weight * loss3 + args.ss_weight * loss4

            ## Backpropagation
            loss.backward()
            optimizer.step()

            ## Record stats
            cls_batch_acc = accuracy(output[nor_index], target, topk=(1,))[0]
            ssp_batch_acc = accuracy(s_simi, aug_target, topk=(1,))[0]
            loss1_record.update(loss1.item(), batch)
            loss2_record.update(loss2.item(), batch)
            loss3_record.update(loss3.item(), len(distill_index_tf))
            loss4_record.update(loss4.item(), len(distill_index_ss))
            cls_acc_record.update(cls_batch_acc.item(), batch)
            ssp_acc_record.update(ssp_batch_acc.item(), 3 * batch)

        ## Log training stats
        logger.add_scalar('train/ce_loss', loss1_record.avg, epoch + 1)
        logger.add_scalar('train/kd_loss', loss2_record.avg, epoch + 1)
        logger.add_scalar('train/tf_loss', loss3_record.avg, epoch + 1)
        logger.add_scalar('train/ss_loss', loss4_record.avg, epoch + 1)
        logger.add_scalar('train/cls_acc', cls_acc_record.avg, epoch + 1)
        logger.add_scalar('train/ss_acc', ssp_acc_record.avg, epoch + 1)

        run_time = time.time() - start
        info = 'student_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ce_loss:{:.3f}\t kd_loss:{:.3f}\t tf_loss:{:.3f}\t ss_loss:{:.3f}\t cls_acc:{:.2f}\t ss_acc:{:.2f}\t'.format(
            epoch + 1, args.epoch, run_time, loss1_record.avg, loss2_record.avg, loss3_record.avg, loss4_record.avg,
            cls_acc_record.avg, ssp_acc_record.avg)
        print(info)

        # =====================
        # Validation (student)
        # =====================
        s_model.eval()
        cls_acc_record = AverageMeter()
        loss_record = AverageMeter()
        start = time.time()
        for x, target in val_loader:
            x = x[:, 0, :, :, :].cuda()
            target = target.cuda()

            with torch.no_grad():
                output, _, _ = s_model(x)
                loss = F.cross_entropy(output, target)

            batch_acc = accuracy(output, target, topk=(1,))[0]
            cls_acc_record.update(batch_acc.item(), x.size(0))
            loss_record.update(loss.item(), x.size(0))

        run_time = time.time() - start
        logger.add_scalar('val/cls_loss', loss_record.avg, epoch + 1)
        logger.add_scalar('val/cls_acc', cls_acc_record.avg, epoch + 1)

        info = 'student_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t cls_loss:{:.3f}\t cls_acc:{:.2f}\n'.format(
            epoch + 1, args.epoch, run_time, loss_record.avg, cls_acc_record.avg)
        print(info)

        scheduler.step()

        ## Save best student model
        if cls_acc_record.avg > best_acc:
            state_dict = dict(epoch=epoch + 1, state_dict=s_model.state_dict(), acc=cls_acc_record.avg)
            name = osp.join(exp_path, 'ckpt/best.pth')
            os.makedirs(osp.dirname(name), exist_ok=True)
            torch.save(state_dict, name)
            best_acc = cls_acc_record.avg

        print('best_acc: {:.2f}'.format(best_acc))

# ---------------------------
# Script entry point
# ---------------------------

## MILEAGE MAY VARY IF RUNNING ON MAC
if __name__ == "__main__":
    import torch
    from torch.multiprocessing import set_start_method

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    main()
