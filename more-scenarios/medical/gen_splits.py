from util.dataset_utils import generate_split, get_patient_ids_from_directory, save_patient_ids, shuffle_split
import yaml


def main():
    cfg = yaml.load(open('configs/ukbb.yaml', "r"), Loader=yaml.Loader)

    # used only once to get all ids
    patient_ids = get_patient_ids_from_directory(cfg['data_root'])
    save_patient_ids(patient_ids, cfg['all_split'], cfg['all_csv'])

    # gen 7 labeled, 63 unlabeled from 70 patients
    generate_split(cfg['labeled_split'], cfg['data_root'], cfg['all_split'], num_patients=7)
    generate_split(cfg['unlabeled_split'], cfg['data_root'], cfg['all_split'], num_patients=63, start_idx=7)
    
    # gen 10 val, 20 test
    generate_split(cfg['val_split'], cfg['data_root'], cfg['all_split'], num_patients=10, start_idx=70, mode='val')
    generate_split(cfg['test_split'], cfg['data_root'], cfg['all_split'], num_patients=20, start_idx=80, mode='test')


if __name__ == '__main__':
    main()
