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
import pprint
import os


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    cfg.update({
        'model_path': f'exp/{cfg["dataset"]}/unimatch/unet/{cfg["split"]}/seed{cfg["seed"]}/best.pth',
        'results_path': f'outputs/results/csv/{cfg["dataset"]}/exp{cfg["exp"]}/seed{cfg["seed"]}/{cfg["control"]}.csv',
        'pred_mask_path': f'outputs/results/imgs/{cfg["dataset"]}/unimatch_{cfg["dataset"]}_{cfg["split"]}_seed{cfg["seed"]}_pred_mask_',
        'test_split_path': f'splits/{cfg["dataset"]}/{cfg["mode"]}/{cfg["control"]}.csv',
    })

    logger.info('{}\n'.format(pprint.pformat({**cfg, **vars(args)})))

    checkpoint = torch.load(cfg['model_path'])
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
        split=cfg['test_split_path'])
    else:
        # ACDC
        test_dataset = ACDCDataset(cfg['dataset'], cfg['data_root'], cfg['mode'])

    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['num_workers'], drop_last=False)
    
    scores_df = eval_model(model, test_loader, cfg, logger, visualize=cfg['visualize'])

    dir_res = os.path.dirname(cfg['results_path'])
    if not os.path.exists(dir_res):
        os.makedirs(dir_res)
    scores_df.to_csv(cfg['results_path'], index=False)


if __name__ == '__main__':
    main()
