from dataset.acdc import ACDCDataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model.unet import UNet
import logging
from util.utils import init_log
import yaml
from util.inference_utils import compute_dice, save_pred_mask, init_scores_df, save_scores


def eval_model(model, dataloader, cfg, logger):
    model.eval()
    model = model.cuda()
    scores_df = init_scores_df()

    i = 0
    with torch.no_grad():
        for img, mask in dataloader:
            img, mask = img.cuda(), mask.cuda()
            # TODO return patient_id, frame with each batch item
            patient_id = "11"
            if i == 0:
                frame = "sa_ED"
            if i == 1:
                frame = "sa_ES"

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
            
            # save og_img, mask, pred
            save_pred_mask(og_img, mask, pred, patient_id, frame, cfg)
            
            # log and save results
            scores_df = save_scores(scores_df, dice_class, dice_mean, patient_id, frame, logger, cfg)
            
            if i == 1:
                break
            i += 1

    return scores_df
    

def main():
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    cfg = yaml.load(open('configs/acdc.yaml', "r"), Loader=yaml.Loader)

    checkpoint = torch.load('exp/acdc/unimatch/unet/7/best.pth')
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model = UNet(in_chns=1, class_num=4)
    model.load_state_dict(checkpoint)

    test_dataset = ACDCDataset(cfg['dataset'], cfg['data_root'], 'test')
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['num_workers'], drop_last=False)
    
    scores = eval_model(model, test_loader, cfg, logger)
    scores.to_csv(cfg['results_path'], index=False)


if __name__ == '__main__':
    main()
