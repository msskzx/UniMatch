from dataset.acdc import ACDCDataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model.unet import UNet
import logging
from util.utils import init_log
import yaml
import pandas as pd
from util.classes import CLASSES
import cv2
import numpy as np


def compute_dice(pred, mask, num_classes=4, epsilon=1e-9):
    # TODO why not same shape like input
    dice_class = [0] * (num_classes - 1)

    for cls in range(1, num_classes):
        pred_class = (pred == cls).float()
        target_class = (mask == cls).float()

        intersection = torch.sum(pred_class * target_class).item()
        union = torch.sum(pred_class).item() + torch.sum(target_class).item()
        
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_class[cls - 1] = dice * 100

    dice_mean = sum(dice_class) / len(dice_class)
    return dice_class, dice_mean


def class_to_intensity(img):
    class_to_intensity = {
        0: 0,   # Background
        1: 85,  # Class 1
        2: 170, # Class 2
        3: 255  # Class 3
    }
    return np.vectorize(lambda x: class_to_intensity[x])(img)
    

def save_pred_mask(image, mask, pred, cfg):
    image = image.cpu().numpy()
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype('uint8')

    mask = class_to_intensity(mask.cpu().numpy())
    pred = class_to_intensity(pred.cpu().numpy())

    # Concatenate the image, gt mask, and prediction horizontally
    concat_img = np.concatenate((image, mask, pred), axis=1)
    cv2.imwrite(cfg['pred_mask_path'], concat_img)


def eval_model(model, dataloader, cfg, logger):
    model.eval()
    model = model.cuda()
    scores = pd.DataFrame(columns=['dice_mean', 'dice_lv', 'dice_rv', 'dice_myo'])
    i = 0
    with torch.no_grad():
        for img, mask in dataloader:
            img, mask = img.cuda(), mask.cuda()
            og_img = img

            h, w = img.shape[-2:]
            img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

            # a batch of number slices in the image
            img = img.permute(1, 0, 2, 3)

            pred = model(img)
            
            pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
            pred = pred.argmax(dim=1).unsqueeze(0)
        
            # compute dice
            dice_class, dice_mean = compute_dice(pred, mask)
            
            logger.info(og_img.shape)
            logger.info(mask.shape)
            logger.info(pred.shape)
            
            # adjust shape
            og_img = og_img.squeeze()[2]
            mask = mask.squeeze()[2]
            pred = pred.squeeze()[2]

            logger.info(og_img.shape)
            logger.info(mask.shape)
            logger.info(pred.shape)

            # save og_img, mask, pred
            save_pred_mask(og_img, mask, pred, cfg)
            
            # log and save results
            for (cls_idx, dice) in enumerate(dice_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                                '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
            logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(dice_mean))

            scores.loc[len(scores)] = {
                    'dice_mean': dice_mean,
                    'dice_lv': dice_class[2],
                    'dice_rv': dice_class[0],
                    'dice_myo': dice_class[1]}
            
            if i == 0:
                break
            i += 1


    return scores
    

def main():
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    cfg = yaml.load(open('configs/acdc.yaml', "r"), Loader=yaml.Loader)

    checkpoint = torch.load('exp/acdc/unimatch/unet/7/best.pth')
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model = UNet(in_chns=1, class_num=4)
    model.load_state_dict(checkpoint)

    test_dataset = ACDCDataset(cfg['dataset'], cfg['data_root'], 'val')
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['num_workers'], drop_last=False)
    
    scores = eval_model(model, test_loader, cfg, logger)
    scores.to_csv(cfg['results_path'], index=False)


if __name__ == '__main__':
    main()
