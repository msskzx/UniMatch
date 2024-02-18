from util.dataset_utils import generate_split, get_patient_ids_from_directory, save_patient_ids
import yaml


def main():
    cfg = yaml.load(open('configs/ukbb_test.yaml', "r"), Loader=yaml.Loader)

    patient_ids = get_patient_ids_from_directory(cfg['data_root'])

    save_patient_ids(patient_ids, 'splits/ukbb/all.txt')

    # gen 7 labeled, 63 unlabeled from 70 patients
    generate_split('splits/ukbb/7/labeled.txt', num_patients=7)
    generate_split('splits/ukbb/7/unlabeled.txt', num_patients=63, start_idx=7)
    
    # gen 10 val, 20 test
    generate_split('splits/ukbb/val.txt', num_patients=10, start_idx=70, mode='val')
    generate_split('splits/ukbb/test.txt', num_patients=20, start_idx=80, mode='test')

    # gen 14 labeled, 133 unlabeled from 140 patients
    generate_split('splits/ukbb/14/labeled.txt', num_patients=14)
    generate_split('splits/ukbb/14/unlabeled.txt', num_patients=133, start_idx=14)


if __name__ == '__main__':
    main()
