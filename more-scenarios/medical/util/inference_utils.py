import torch
import numpy as np
import cv2
import pandas as pd
from util.classes import CLASSES, MASK
import torch.nn.functional as F

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
        1: 85,  # CLASSES[0]
        2: 170, # CLASSES[1]
        3: 255  # CLASSES[2]
    }
    return np.vectorize(lambda x: class_to_intensity[x])(img)


def save_pred_mask(og_img, og_mask, og_pred, patient_id, frame, cfg):
    for slice_idx in range(og_img.shape[0]):
        img = og_img[slice_idx].cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255
        mask = class_to_intensity(og_mask[slice_idx].cpu().numpy())
        pred = class_to_intensity(og_pred[slice_idx].cpu().numpy())

        # Concatenate the img, gt mask, and prediction horizontally
        concat_img = np.concatenate((img, mask, pred), axis=1)
        cv2.imwrite(f"{cfg['pred_mask_path']}{patient_id}_{frame}_{slice_idx}.png", concat_img)


def init_scores_df():
    return pd.DataFrame(columns=['patient_id', 'frame', 'slice_idx', 'dice_mean', 'dice_rv', 'dice_myo', 'dice_lv'])


def save_scores(scores_df, dice_class, dice_mean, patient_id, frame, logger, cfg):
    for slice_idx, dice_scores_slice in enumerate(dice_class):
        for cls_idx, dice_slice_cls in enumerate(dice_scores_slice):
            # TODO add patient_id, frame, slice_id
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice_slice_cls))
        logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(dice_mean[slice_idx]))

        scores_df.loc[len(scores_df)] = {
                'patient_id': patient_id,
                'frame': frame,
                'slice_idx': slice_idx,
                'dice_mean': dice_mean[slice_idx],
                'dice_rv': dice_class[slice_idx][MASK['rv']-1],
                'dice_myo': dice_class[slice_idx][MASK['myo']-1],
                'dice_lv': dice_class[slice_idx][MASK['lv']-1]}
    
    return scores_df


def eval_model(model, dataloader, cfg, logger):
    model.eval()
    model = model.cuda()
    scores_df = init_scores_df()

    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            img, mask = batch['img'].cuda(), batch['mask'].cuda()
            patient_id = batch['patient_id'][0]
            frame = batch['frame'][0]

            # a batch of number slices in the img
            og_img = img
            img = img.permute(1, 0, 2, 3)

            # interpolate and predict
            h, w = img.shape[-2:]
            img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)
            pred = model(img)
            pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
            pred = pred.argmax(dim=1)

            # adjust shape to compute dice
            mask = mask.squeeze()
            og_img = og_img.squeeze()
            dice_class, dice_mean = compute_dice(pred, mask)
            
            # TODO get slice_idx from imgs instead of creating arbitrary idx

            # save og_img, mask, pred
            # save_pred_mask(og_img, mask, pred, patient_id, frame, cfg)
            
            # log and save results
            scores_df = save_scores(scores_df, dice_class, dice_mean, patient_id, frame, logger, cfg)
            
    return scores_df