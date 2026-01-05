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

import torch.multiprocessing as mp

from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
import torchvision

import torchvision.transforms as transforms
#from torchvision.datasets import CIFAR100
from tensorboardX import SummaryWriter

from utils import AverageMeter, accuracy   # helper functions for tracking metrics
from models import model_dict              # dictionary of models (defined elsewhere)

import random

# Enable cudnn autotuner to select the fastest convolution algorithm
torch.backends.cudnn.benchmark = True

NUM_COCO_CLASSES = 80      # standard COCO object categories
NUM_SSV2_CLASSES = 174     # Something-Something V2 action classes (v2)

import json
from PIL import Image

import os
import subprocess

videos_dir = "./ssv2/videos/20bn-something-something-v2"
out_dir = "./ssv2/frames/train/"

os.makedirs(out_dir, exist_ok=True)

#for video in os.listdir(videos_dir):
#    if not video.endswith(".webm"):
#        continue

#    vid = video.replace(".webm", "")
#    out_path = os.path.join(out_dir, vid)
#    os.makedirs(out_path, exist_ok=True)

#    cmd = [
  #      "ffmpeg",
#        "-i", os.path.join(videos_dir, video),
   #     os.path.join(out_path, "%05d.jpg")
   # ]

   # print("Extracting:", video)
  #  subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class COCOClassification(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.transform = transform

        # Load annotation JSON manually
        with open(annFile, 'r') as f:
            data = json.load(f)

        # Extract categories → contiguous indexing
        categories = data['categories']
        cat_ids = sorted([c['id'] for c in categories])
        self.catid2idx = {cid: i for i, cid in enumerate(cat_ids)}

        # Build mapping: image_id → first annotation's category
        self.image_info = {}
        for img in data['images']:
            self.image_info[img['id']] = {
                "file_name": img["file_name"],
                "category_id": None   # fill later
            }

        # Fill category ids (just use first annotation per image)
        for ann in data['annotations']:
            img_id = ann['image_id']
            if self.image_info[img_id]["category_id"] is None:
                self.image_info[img_id]["category_id"] = ann["category_id"]

        # Convert dict → list
        self.entries = []
        for img_id, info in self.image_info.items():
            file_name = info["file_name"]
            cat_id = info["category_id"]
            if cat_id is None:
                # skip images without labels
                continue
            label = self.catid2idx[cat_id]
            self.entries.append((file_name, label))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        file_name, label = self.entries[idx]

        # Load the image
        path = os.path.join(self.root, file_name)
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label



import json
import os
import random
import torchvision
from torch.utils.data import Dataset

class SomethingSomethingFrameDataset(Dataset):
    """
    Frame-based SSv2 dataset built from JSON list:
    [
      {"id": "78687", "template": "...", ...},
      ...
    ]
    Uses 'template' as class; maps unique templates to indices.
    """
    def __init__(self, json_file, frame_root, transform=None, template2idx=None):
        self.frame_root = frame_root
        self.transform = transform

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected SSv2 JSON to be a list of dicts.")

        # Build template mapping if not provided (train)
        if template2idx is None:
            templates = sorted({item["template"] for item in data if "template" in item})
            self.template2idx = {t: i for i, t in enumerate(templates)}
        else:
            self.template2idx = template2idx

        self.samples = []
        for item in data:
            vid = item.get("id")
            templ = item.get("template")
            if vid is None or templ is None:
                continue
            if templ not in self.template2idx:
                continue

            label = self.template2idx[templ]
            frame_dir = os.path.join(frame_root, str(vid))
            if not os.path.isdir(frame_dir):
                continue

            frames = [f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")]
            if not frames:
                continue

            self.samples.append((frame_dir, frames, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid SSv2 samples found under {frame_root}. "
                "Check that video ids in JSON match folder names."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_dir, frames, label = self.samples[idx]
        fname = random.choice(frames)
        path = os.path.join(frame_dir, fname)

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

class MultiHeadTeacher(nn.Module):
    def __init__(self, num_coco_classes, num_ssv2_classes):
        super().__init__()

        # backbone – ImageNet-style
        backbone = torchvision.models.resnet50(weights=None)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # we’ll provide our own heads
        self.backbone = backbone

        # heads
        self.head_coco = nn.Linear(in_dim, num_coco_classes)
        self.head_ssv2 = nn.Linear(in_dim, num_ssv2_classes)

    def forward(self, x, dataset: str):
        """
        dataset: 'coco' or 'ssv2'
        """
        feat = self.backbone(x)  # [B, in_dim]
        if dataset == 'coco':
            return self.head_coco(feat)
        elif dataset == 'ssv2':
            return self.head_ssv2(feat)
        else:
            raise ValueError(f"Unknown dataset tag: {dataset}")


def main():
    parser = argparse.ArgumentParser(description='Train multi-head teacher on COCO + SSv2.')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--milestones', type=int, nargs='+', default=[30, 40])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    exp_name = f'teacher_multih_{args.seed}'
    exp_path = f'./experiments/{exp_name}'
    os.makedirs(exp_path, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # <-- converts PIL OR ndarray to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ---------------------------
    # Datasets & loaders
    # ---------------------------
    coco_train = COCOClassification(
        root="./coco/train2017",
        annFile="./coco/annotations/instances_train2017.json",
        transform=transform_train
    )
    coco_loader = DataLoader(
        coco_train, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )

    ssv2_train = SomethingSomethingFrameDataset(
        json_file="./ssv2/train.json",
        frame_root="./ssv2/frames/train",
        transform=transform_train
    )
    NUM_SSV2_CLASSES = len(ssv2_train.template2idx)

    ssv2_val = SomethingSomethingFrameDataset(
        json_file="./ssv2/validation.json",  # if you have it
        frame_root="./ssv2/frames/train",
        transform=transform_train,
        template2idx=ssv2_train.template2idx
    )

    print("SSv2 template classes:", NUM_SSV2_CLASSES)

    ssv2_loader = DataLoader(
        ssv2_train, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )

    coco_iter = iter(coco_loader)
    ssv2_iter = iter(ssv2_loader)

    # ---------------------------
    # COCO Validation Dataset
    # ---------------------------
    coco_val = COCOClassification(
        root="./coco/val2017",
        annFile="./coco/annotations/instances_val2017.json",
        transform=transform_train  # you can switch to transform_test
    )

    coco_val_loader = DataLoader(
        coco_val, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    # ---------------------------
    # SSv2 Validation Dataset
    # ---------------------------
    ssv2_loader = DataLoader(
        ssv2_train, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )

    ssv2_val_loader = DataLoader(
        ssv2_val, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    # ---------------------------
    # Model, optimizer, scheduler
    # ---------------------------
    model = MultiHeadTeacher(NUM_COCO_CLASSES, NUM_SSV2_CLASSES).cuda()

    start_epoch = 0

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"=> Resumed from {args.resume} (epoch {start_epoch})")


    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    logger = SummaryWriter(osp.join(exp_path, 'events'))
    best_coco_acc = -1.0  # you can extend this for ssv2 if you later add val

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(start_epoch, args.epoch):
        model.train()
        loss_record = AverageMeter()
        acc_record = AverageMeter()

        start = time.time()

        # define number of steps per epoch
        steps_per_epoch = max(len(coco_loader), len(ssv2_loader))

        for step in range(steps_per_epoch):
            # randomly choose dataset for this step
            if random.random() < 0.5:
                dataset_name = 'coco'
                try:
                    x, target = next(coco_iter)
                except StopIteration:
                    coco_iter = iter(coco_loader)
                    x, target = next(coco_iter)
            else:
                dataset_name = 'ssv2'
                try:
                    x, target = next(ssv2_iter)
                except StopIteration:
                    ssv2_iter = iter(ssv2_loader)
                    x, target = next(ssv2_iter)

            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            optimizer.zero_grad()
            output = model(x, dataset=dataset_name)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            batch_acc = accuracy(output, target, topk=(1,))[0]
            loss_record.update(loss.item(), x.size(0))
            acc_record.update(batch_acc.item(), x.size(0))

        run_time = time.time() - start

        logger.add_scalar('train/loss', loss_record.avg, epoch + 1)
        logger.add_scalar('train/acc', acc_record.avg, epoch + 1)

        info = f'train_Epoch:{epoch+1:03d}/{args.epoch:03d}\t' \
               f'run_time:{run_time:.3f}\t' \
               f'loss:{loss_record.avg:.3f}\t' \
               f'acc:{acc_record.avg:.2f}'
        print(info)

        scheduler.step()

        # ---------------------------
        # VALIDATION LOOP
        # ---------------------------
        model.eval()

        val_coco_loss = AverageMeter()
        val_coco_acc = AverageMeter()

        val_ssv2_loss = AverageMeter()
        val_ssv2_acc = AverageMeter()

        with torch.no_grad():

            # ---- COCO Validation ----
            for x, target in coco_val_loader:
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                output = model(x, dataset="coco")
                loss = F.cross_entropy(output, target)

                acc = accuracy(output, target, topk=(1,))[0]

                val_coco_loss.update(loss.item(), x.size(0))
                val_coco_acc.update(acc.item(), x.size(0))

            # ---- SSv2 Validation ----
            for x, target in ssv2_val_loader:
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                output = model(x, dataset="ssv2")
                loss = F.cross_entropy(output, target)

                acc = accuracy(output, target, topk=(1,))[0]

                val_ssv2_loss.update(loss.item(), x.size(0))
                val_ssv2_acc.update(acc.item(), x.size(0))

        # ---- Print Validation Results ----

        print(
            f"[VAL EPOCH {epoch + 1}] "
            f"COCO Acc: {val_coco_acc.avg:.2f}%  "
            f"SSv2 Acc: {val_ssv2_acc.avg:.2f}%"
        )

        # ---- TensorBoard Logs ----
        logger.add_scalar('val/coco_acc', val_coco_acc.avg, epoch + 1)
        logger.add_scalar('val/ssv2_acc', val_ssv2_acc.avg, epoch + 1)
        logger.add_scalar('val/coco_loss', val_coco_loss.avg, epoch + 1)
        logger.add_scalar('val/ssv2_loss', val_ssv2_loss.avg, epoch + 1)

        # checkpoint
        state_dict = dict(epoch=epoch+1, state_dict=model.state_dict())
        ckpt_dir = osp.join(exp_path, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(state_dict, osp.join(ckpt_dir, f'{epoch+1:03d}.pth'))

    print("Training done.")

# ---------------------------
# Script entry point
# ---------------------------

## MILEAGE MAY VARY IF RUNNING ON MAC
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import torch
    from torch.multiprocessing import set_start_method

    # For DataLoader multiprocessing compatibility on some systems
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    main()
