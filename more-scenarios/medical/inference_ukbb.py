import matplotlib.pyplot as plt
from dataset.ukbb import UKBBDataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model.unet import UNet
import logging
from util.utils import init_log

def eval_model(model, dataloader):
    model.eval()
    model = model.cuda()
    scores = pd.DataFrame(columns=['dice_mean'])
    dice_class = [0] * 3

    for img, mask in dataloader:
        img, mask = img.cuda(), mask.cuda()

        with torch.no_grad():
            pred = model(img)

        pred = pred.argmax(dim=1).unsqueeze(0)

        # compute dice
        for cls in range(1, 4):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls-1] += 2.0 * inter / union

        dice_class = [dice * 100.0 / len(dataloader) for dice in dice_class]
        dice_mean = sum(dice_class) / len(dice_class)

        scores.loc[len(scores)] = dice_mean

    return scores


def eval_model_alt(model, dataloader):
    model.eval()
    dice_class = [0] * (cfg['nclass'] - 1)
    scores = pd.DataFrame(columns=['dice_mean'])

    for batch_idx, batch in enumerate(test_loader):
        image, mask = batch['image'].cuda(), batch['mask'].cuda()

        image = image.permute(1, 0, 2, 3)
        
        with torch.no_grad():
            pred = model(image)
        pred = pred.argmax(dim=1).unsqueeze(0)

        for cls in range(1, nclass):
            inter = ((pred == cls) * (mask == cls)).sum().item()
            union = (pred == cls).sum().item() + (mask == cls).sum().item()
            dice_class[cls-1] += 2.0 * inter / union

        dice_class = [dice * 100.0 / len(test_loader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)
        scores.loc[len(scores)] = dice_mean

        for (cls_idx, dice) in enumerate(dice_class):
            logger.info('***** Evaluation ***** >>>> Class [{:}] Dice: '
                        '{:.2f}'.format(cls_idx, dice))
        logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice))
    
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
        patient_ids=patient_ids)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    scores = eval_model(model, test_loader)
    scores.to_csv(cfg['exp'], index=False)


if __name__ == '__main__':
    main()
