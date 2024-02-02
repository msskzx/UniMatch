import matplotlib.pyplot as plt
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

def eval_model(model, dataloader, cfg, logger):
    model.eval()
    model = model.cuda()
    scores = pd.DataFrame(columns=['patient_id', 'dice_mean', 'dice_lv', 'dice_rv', 'dice_myo'])
    dice_class = [0] * (cfg['nclass'] - 1)

    for batch_idx, batch in enumerate(dataloader):
        image, mask = batch['image'].cuda(), batch['mask'].cuda()
        logger.info(image.shape)
        h, w = image.shape[-2:]
        image = F.interpolate(image, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

        image = image.permute(1, 0, 2, 3)
        logger.info(image.shape)
        
        with torch.no_grad():
            pred = model(image)

        pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
        pred = pred.argmax(dim=1).unsqueeze(0)

        # compute dice
        for cls in range(1, cfg['nclass']):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls-1] += 2.0 * inter / union

        dice_class = [dice * 100.0 / len(dataloader) for dice in dice_class]
        dice_mean = sum(dice_class) / len(dice_class)
        for (cls_idx, dice) in enumerate(dice_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
        logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(dice_mean))

        scores.loc[len(scores)] = {
                'patient_id': batch['patient_id'],
                'dice_mean': dice_mean,
                'dice_lv': dice_class[2],
                'dice_rv': dice_class[0],
                'dice_myo': dice_class[1]
            }

    return scores
    

def main():
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    checkpoint = torch.load('exp/acdc/unimatch/unet/7/best.pth')
    model = UNet(in_chns=1, class_num=4)
    model_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(model_checkpoint)

    cfg = yaml.load(open('configs/ukbb.yaml', "r"), Loader=yaml.Loader)
    patient_ids = ["5733285"]

    test_dataset = UKBBDataset(
        name=cfg['dataset'],
        root_dir=cfg['data_root'],
        mode=cfg['mode'],
        crop_size=cfg['crop_size'],
        patient_ids=patient_ids
    )

    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    
    scores = eval_model(model, test_loader, cfg, logger)
    scores.to_csv(cfg['results_path'], index=False)


if __name__ == '__main__':
    main()
