from dataset.ukbb import UKBBDataset
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
    dice_class = [0] * (num_classes - 1)
    for cls in range(num_classes):
        pred_class = (pred == cls).float()
        target_class = (mask == cls).float()
        intersection = torch.sum(pred_class * target_class)
        union = torch.sum(pred_class) + torch.sum(target_class)
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_class[cls - 1] = dice.item() * 100
    dice_mean = sum(dice_class) / num_classes
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
    mask = class_to_intensity(mask.cpu().numpy())
    pred = class_to_intensity(pred.cpu().numpy())
    # Concatenate the image, gt mask, and prediction horizontally
    concat_img = np.concatenate((image, mask, pred), axis=1)
    cv2.imwrite(cfg['pred_mask_path'], concat_img)


def eval_model(model, dataloader, cfg, logger):
    model.eval()
    model = model.cuda()
    scores = pd.DataFrame(columns=['patient_id', 'dice_mean', 'dice_lv', 'dice_rv', 'dice_myo'])

    for batch_idx, batch in enumerate(dataloader):
        # interpolate and predict
        image, mask = batch['image'].cuda(), batch['mask'].cuda()

        input_image = image
        h, w = image.shape[-2:]
        image = F.interpolate(image, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)
        image = image.permute(1, 0, 2, 3)
        with torch.no_grad():
            pred = model(image)
        pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
        pred = pred.argmax(dim=1)
        
        # adjust shape
        input_image = input_image.permute(2, 3, 1, 0).squeeze()
        mask = mask.permute(2, 3, 1, 0).squeeze()
        pred = pred.permute(1, 2, 0).squeeze()

        # compute dice
        dice_class, dice_mean = compute_dice(pred, mask)
        
        # save input_image, mask, pred
        save_pred_mask(input_image, mask, pred, cfg)

        # log and save results
        for (cls_idx, dice) in enumerate(dice_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
        logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(dice_mean))

        scores.loc[len(scores)] = {
                'patient_id': batch['patient_id'][0],
                'dice_mean': dice_mean,
                'dice_lv': dice_class[2],
                'dice_rv': dice_class[0],
                'dice_myo': dice_class[1]}

    return scores
    

def main():
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    cfg = yaml.load(open('configs/ukbb.yaml', "r"), Loader=yaml.Loader)
    model = None

    if cfg['model'] == 'unimatch':
        checkpoint = torch.load('exp/acdc/unimatch/unet/7/best.pth')
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model = UNet(in_chns=1, class_num=4)
        model.load_state_dict(checkpoint)
    elif cfg['model'] == 'fct':
        model = torch.load('model/fct.model')

    patient_ids = ["5733285"]

    test_dataset = UKBBDataset(
        name=cfg['dataset'],
        root_dir=cfg['data_root'],
        mode=cfg['mode'],
        crop_size=cfg['crop_size'],
        patient_ids=patient_ids)

    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    
    scores = eval_model(model, test_loader, cfg, logger)
    scores.to_csv(cfg['results_path'], index=False)


if __name__ == '__main__':
    main()
