import torch
from torch.utils.data import DataLoader
import logging
import yaml
import argparse
import pprint
import os
from transformers import BertModel, BertTokenizer

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
parser.add_argument('--dataset', type=str, required=True)


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(f'configs/{args.dataset}/test/exp{args.exp}/{args.control}.yaml', "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    test_split_file = cfg['control']
    if cfg['task'] in ['multi_task', 'multi_modal', 'seg_only_mid_slices']:
        test_split_file += '_mt'

    cfg.update({
        'model_path': f'exp/{cfg["dataset"]}/{args.method}/{cfg["seg_model"]}/exp{args.exp}/seed{args.seed}/{cfg["task"]}/best.pth',
        'results_path': f'outputs/results/csv/{cfg["dataset"]}/{args.method}/{cfg["seg_model"]}/exp{args.exp}/seed{args.seed}/{cfg["task"]}/{cfg["control"]}.csv',
        'pred_mask_path': f'outputs/results/imgs//{cfg["dataset"]}/{args.method}/{cfg["seg_model"]}/exp{args.exp}/seed{args.seed}/{cfg["task"]}/',
        'test_split_path': f'splits/{cfg["dataset"]}/{cfg["mode"]}/{test_split_file}.csv',
    })

    os.makedirs(f'outputs/results/imgs//{cfg["dataset"]}/{args.method}/{cfg["seg_model"]}/exp{args.exp}/seed{args.seed}/{cfg["task"]}/', exist_ok=True)

    logger.info('{}\n'.format(pprint.pformat({**cfg, **vars(args)})))

    checkpoint = torch.load(cfg['model_path'])
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    
    label_embeddings = None
    if cfg['task'] == 'multi_task':
        model = UNetMultiTask(in_chns=1, nclass=cfg['nclass'], nclass_classif=cfg['nclass_classif'])
    elif cfg['task'] == 'multi_modal':
        # Convert label text to BERT embeddings
        labels = ['white', 'asian', 'black']
        bert_model_name='bert-base-uncased'
        bert_model = BertModel.from_pretrained(bert_model_name)
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_embedding_dim = bert_model.config.hidden_size

        model = UNetMultiModal(in_chns=1, nclass=cfg['nclass'], bert_embedding_dim=bert_embedding_dim)

        label_embeddings = {}
        for label in labels:
            inputs = tokenizer(label, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)

            label_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            label_embeddings[label] = label_embedding
    else:
        model = UNet(in_chns=1, class_num=4)

    
    model.load_state_dict(checkpoint)

    if cfg['dataset'] == 'ukbb':
        test_dataset = UKBBDataset(
        name=cfg['dataset'],
        root_dir=cfg['data_root'],
        mode=cfg['mode'],
        crop_size=cfg['crop_size'],
        split=cfg['test_split_path'],
        task=cfg['task']
        )
    else:
        test_dataset = ACDCDataset(cfg['dataset'], cfg['data_root'], cfg['mode'])

    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['num_workers'], drop_last=False)
    
    scores_df = eval_model(model, test_loader, cfg, logger, label_embeddings=label_embeddings, visualize=cfg['visualize'])

    dir_res = os.path.dirname(cfg['results_path'])
    os.makedirs(dir_res, exist_ok=True)
    scores_df.to_csv(cfg['results_path'], index=False)


if __name__ == '__main__':
    main()
