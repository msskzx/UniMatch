from util.dataset_utils import generate_split
import yaml


def main():
    cfg = yaml.load(open('configs/ukbb.yaml', "r"), Loader=yaml.Loader)
    generate_split(input_file='ukbb/val.csv', output_file='splits/ukbb/val.csv', mode='val', shuffle=True)
    generate_split(input_file='ukbb/test_sex_ctrl.csv', output_file='splits/ukbb/test_sex_ctrl.csv', mode='test', shuffle=False)
    generate_split(input_file='ukbb/test_ethn_ctrl.csv', output_file='splits/ukbb/test_ethn_ctrl.csv', mode='test', shuffle=False)
    generate_split(input_file='ukbb/train.csv', output_file='splits/ukbb/train.csv', mode='train', cfg=cfg, shuffle=True)


if __name__ == '__main__':
    main()
