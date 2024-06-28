import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from model.unet import UNet
from util.classes import CLASSES
from util.utils import AverageMeter, count_params, init_log, DiceLoss
from util.dist_helper import setup_distributed
from dataset.ukbb import UKBBDataset


parser = argparse.ArgumentParser(description='Fully Supervised UNet on UKBB')
parser.add_argument('--seed', type=str, required=True)
parser.add_argument('--exp', type=str, required=True)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--local_rank', default=0, type=int)


def main():
    args = parser.parse_args()

    method='supervised'
    seg_model='unet'
    cfg = yaml.load(open(f'configs/ukbb/train/exp{args.exp}/config.yaml', "r"), Loader=yaml.Loader)
    save_path = f'exp/{cfg["dataset"]}/{method}/{seg_model}/exp{args.exp}/seed{args.seed}'

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(save_path)
        
        os.makedirs(save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = UNet(in_chns=1, class_num=cfg['nclass'])
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.0001)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg['nclass'])
    
    trainset = UKBBDataset(
        name=cfg['dataset'],
        root_dir=cfg['data_root'],
        mode='train_l',
        crop_size=cfg['crop_size'],
        split=f'splits/{cfg["dataset"]}/exp{args.exp}/{cfg["split"]}/seed{args.seed}/train.csv'
    )
    
    valset = UKBBDataset(
        name=cfg['dataset'],
        root_dir=cfg['data_root'],
        mode='val',
        crop_size=cfg['crop_size'],
        split=f'splits/{cfg["dataset"]}/exp{args.exp}/{cfg["split"]}/seed{args.seed}/val.csv'
    )

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask, ethn) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            loss = (criterion_ce(pred, mask) + criterion_dice(pred.softmax(dim=1), mask.unsqueeze(1).float())) / 2.0
            
            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)
            
            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        model.eval()
        dice_class = [0] * 3
        
        with torch.no_grad():
            for _, (img, mask, ethn) in enumerate(valloader):
                img, mask = img.cuda(), mask.cuda()

                h, w = img.shape[-2:]
                img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

                img = img.permute(1, 0, 2, 3)
                
                pred = model(img)
                
                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
                pred = pred.argmax(dim=1)

                mask = mask.squeeze()
                epsilon=1e-9

                for cls in range(1, cfg['nclass']):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls-1] += (2.0 * inter + epsilon) / (union + epsilon) * 100.0

        dice_class = [dice / len(valloader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)
        
        if rank == 0:
            for (cls_idx, dice) in enumerate(dice_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
            logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice))
            
            writer.add_scalar('eval/MeanDice', mean_dice, epoch)
            for i, dice in enumerate(dice_class):
                writer.add_scalar('eval/%s_dice' % (CLASSES[cfg['dataset']][i]), dice, epoch)

        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(save_path, 'best.pth'))


if __name__ == '__main__':
    main()
