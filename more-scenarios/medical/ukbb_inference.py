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
    num_slices = pred.shape[0]
    dice_class = np.zeros((num_slices, num_classes - 1))
    dice_mean = np.zeros(num_slices)

    for slice in range(num_slices):
        for cls in range(1, num_classes):
            pred_class = (pred[slice] == cls).float()
            target_class = (mask[slice] == cls).float()

            intersection = torch.sum(pred_class * target_class).item()
            union = torch.sum(pred_class).item() + torch.sum(target_class).item()
            
            dice = (2. * intersection + epsilon) / (union + epsilon)
            dice_class[slice][cls - 1] = dice * 100

        dice_mean[slice] = sum(dice_class[slice]) / (num_classes - 1)
    return dice_class, dice_mean


def class_to_intensity(img):
    class_to_intensity = {
        0: 0,   # Background
        1: 85,  # Class 1
        2: 170, # Class 2
        3: 255  # Class 3
    }
    return np.vectorize(lambda x: class_to_intensity[x])(img)
    

def save_pred_mask(img, mask, pred, cfg, patient_id = 1, slice_idx=1):
    img = img.cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min()) * 255
    mask = class_to_intensity(mask.cpu().numpy())
    pred = class_to_intensity(pred.cpu().numpy())

    # Concatenate the img, gt mask, and prediction horizontally
    concat_img = np.concatenate((img, mask, pred), axis=1)
    cv2.imwrite(f"{cfg['pred_mask_path']}{patient_id}_{slice_idx}.png", concat_img)
        

def eval_model(model, dataloader, cfg, logger):
    model.eval()
    model = model.cuda()
    scores = pd.DataFrame(columns=['patient_id', 'slice_id', 'dice_mean', 'dice_lv', 'dice_rv', 'dice_myo'])
    
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            img, mask = batch['image'].cuda(), batch['mask'].cuda()

            # a batch of number slices in the img
            img = img.permute(3, 0, 1, 2)
            mask = mask.permute(0, 3, 1, 2)
            og_img = img.permute(1, 0, 2, 3)
            
            # interpolate and predict
            h, w = img.shape[-2:]
            img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

            pred = model(img)
            
            pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
            pred = pred.argmax(dim=1)

            # adjust shape
            mask = mask.squeeze()

            # compute dice
            dice_class, dice_mean = compute_dice(pred, mask)

            og_img = og_img.squeeze()

            # save og_img, mask, pred
            for slice_idx,_ in enumerate(og_img):
                save_pred_mask(og_img[slice_idx], mask[slice_idx], pred[slice_idx], cfg, batch['patient_id'][0], slice_idx)

            # log and save results
            for slice_idx, dice_scores_slice in enumerate(dice_class):
                print(f"Dice scores for slice {slice_idx + 1}:")
                for cls_idx, dice_slice_cls in enumerate(dice_scores_slice):
                    logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                                    '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice_slice_cls))
                logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(dice_mean[slice_idx]))

                # TODO save class name and mask index globally and fetch that instead.
                scores.loc[len(scores)] = {
                        'patient_id': batch['patient_id'][0],
                        'slice_id': slice_idx,
                        'dice_mean': dice_mean[slice_idx],
                        'dice_lv': dice_class[slice_idx][0],
                        'dice_rv': dice_class[slice_idx][2],
                        'dice_myo': dice_class[slice_idx][1]}

    return scores
    

def main():
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    cfg = yaml.load(open('configs/ukbb.yaml', "r"), Loader=yaml.Loader)
    model = None

    checkpoint = torch.load('exp/acdc/unimatch/unet/7/best.pth')
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model = UNet(in_chns=1, class_num=4)
    model.load_state_dict(checkpoint)

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
