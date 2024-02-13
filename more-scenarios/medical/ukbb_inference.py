from dataset.ukbb import UKBBDataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model.unet import UNet
import logging
from util.utils import init_log
import yaml
from util.inference_utils import compute_dice, save_pred_mask, init_scores_df, save_scores

# TODO 1. get test split
# TODO 2. get patients info
# TODO 3. analysis

def eval_model(model, dataloader, cfg, logger):
    model.eval()
    model = model.cuda()
    scores_df = init_scores_df()
    
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            # TODO handle batch size
            img, mask = batch['image'].cuda(), batch['mask'].cuda()
            patient_id = batch['patient_id'][0]
            frame = batch['frame'][0]

            # a batch of number slices in the img
            img = img.permute(1, 0, 2, 3)
            og_img = img
            
            # interpolate and predict
            h, w = img.shape[-2:]
            img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

            pred = model(img)
            
            pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
            pred = pred.argmax(dim=1)

            # adjust shape to compute dice
            mask = mask.squeeze()
            og_img = og_img.squeeze()

            # compute dice
            dice_class, dice_mean = compute_dice(pred, mask)

            # save og_img, mask, pred
            save_pred_mask(og_img, mask, pred, patient_id, frame, cfg)

            # log and save results
            scores_df = save_scores(scores_df, dice_class, dice_mean, patient_id, frame, logger, cfg)

    return scores_df
    

def main():
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    cfg = yaml.load(open('configs/ukbb.yaml', "r"), Loader=yaml.Loader)
    model = None

    checkpoint = torch.load('exp/acdc/unimatch/unet/7/best.pth')
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model = UNet(in_chns=1, class_num=4)
    model.load_state_dict(checkpoint)

    patient_ids_frames = [("5733285", "sa_ES"), ("5733285", "sa_ED")]

    test_dataset = UKBBDataset(
        name=cfg['dataset'],
        root_dir=cfg['data_root'],
        mode=cfg['mode'],
        crop_size=cfg['crop_size'],
        patient_ids_frames=patient_ids_frames)

    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    
    scores = eval_model(model, test_loader, cfg, logger)
    scores.to_csv(cfg['results_path'], index=False)


if __name__ == '__main__':
    main()
