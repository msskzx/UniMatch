import torch
from torch.utils.data import DataLoader
import logging
import yaml
import argparse
import pprint
import os

from util.utils import init_log
from util.inference_utils import eval_model
from dataset.acdc import ACDCDataset
from dataset.ukbb import UKBBDataset
from model.unet import UNet
from model.unet_multi_task import UNetMultiTask
from model.unet_multi_modal import UNetMultiModal


parser = argparse.ArgumentParser(description='Infereing using the pretrained models')
parser.add_argument('--control', type=str, required=True)
parser.add_argument('--seed', type=str, required=True)
parser.add_argument('--exp', type=str, required=True)
parser.add_argument('--method', type=str, required=True)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(f'configs/ukbb/test/exp{args.exp}/{args.control}.yaml', "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    model_path = f'exp/{cfg["dataset"]}/{args.method}/unet/exp{args.exp}/seed{args.seed}'
    results_path = f'outputs/results/csv/{cfg["dataset"]}/{args.method}/unet/exp{args.exp}/seed{args.seed}'
    if cfg['task'] == 'multi_task':
        model_path += '/multi_task'
        results_path += '/multi_task'

    # TODO test split path, test split _mt (add missing fields)
    cfg.update({
        'model_path': f'{model_path}/best.pth',
        'results_path': f'{results_path}/{cfg["control"]}.csv',
        'pred_mask_path': f'outputs/results/imgs/{cfg["dataset"]}/{args.method}/{args.exp}/{cfg["split"]}/seed{args.seed}',
        'test_split_path': f'splits/{cfg["dataset"]}/{cfg["mode"]}/{cfg["control"]}.csv',
    })

    logger.info('{}\n'.format(pprint.pformat({**cfg, **vars(args)})))

    checkpoint = torch.load(cfg['model_path'])
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}

    # fully supervised or semi-supervised
    if cfg['task'] == 'multi_task':
        model = UNetMultiTask(in_chns=1, seg_nclass=cfg['nclass'], classif_nclass=3)
    elif cfg['task'] == 'multi_modal':
        model = UNetMultiModal(in_chns=1, seg_nclass=cfg['nclass'], classif_nclass=3)
    else:
        model = UNet(in_chns=1, class_num=4)

    
    # TODO incorporate the label

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
    os.makedirs(dir_res, exist_ok=True)
    scores_df.to_csv(cfg['results_path'], index=False)


if __name__ == '__main__':
    main()
