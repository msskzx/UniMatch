from dataset.acdc import ACDCDataset
from dataset.ukbb import UKBBDataset
import torch
from torch.utils.data import DataLoader
from model.unet import UNet
import logging
from util.utils import init_log
import yaml
from util.inference_utils import eval_model
import argparse


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    logger.info(cfg['model_desc'])
    logger.info(cfg['data_desc'])

    checkpoint = torch.load(cfg['model'])
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model = UNet(in_chns=1, class_num=4)
    model.load_state_dict(checkpoint)

    if cfg['dataset'] == 'ukbb':
        # UKBB
        test_dataset = UKBBDataset(
        name=cfg['dataset'],
        root_dir=cfg['data_root'],
        mode=cfg['mode'],
        crop_size=cfg['crop_size'],
        split=cfg['test_split'])
    else:
        # ACDC
        test_dataset = ACDCDataset(cfg['dataset'], cfg['data_root'], cfg['mode'])

    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['num_workers'], drop_last=False)
    
    scores_df = eval_model(model, test_loader, cfg, logger)
    scores_df.to_csv(cfg['results_path'], index=False)


if __name__ == '__main__':
    main()
