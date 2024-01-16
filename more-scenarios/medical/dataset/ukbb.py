import os
from torch.utils.data import Dataset
import nibabel as nib


class UKBBDataset(Dataset):
    def __init__(self, name, root_dir, mode, patient_ids, size=None, id_path=None, nsample=None, transform=None):
        self.name = name
        self.root_dir = root_dir
        self.mode = mode
        self.size = size
        self.patient_ids = patient_ids

    def __getitem__(self, item):
        patient_id = self.patient_ids[item]
        return self.get_patient_images(patient_id)

    def __len__(self):
        return len(self.patient_ids)

    def get_patient_images(self, patient_id):
        # Load original images (es.nii.gz and ed.nii.gz) for the current patient
        sa_es_path = os.path.join(self.root_dir, patient_id, "sa_ES.nii.gz")
        sa_ed_path = os.path.join(self.root_dir, patient_id, "sa_ED.nii.gz")
        sa_es = nib.load(sa_es_path).get_fdata()
        sa_ed = nib.load(sa_ed_path).get_fdata()

        # Load segmentation images (seg_es.nii.gz and seg_ed.nii.gz) for the current patient
        seg_sa_es_path = os.path.join(self.root_dir, patient_id, "seg_sa_ED.nii.gz")
        seg_sa_ed_path = os.path.join(self.root_dir, patient_id, "seg_sa_ES.nii.gz")
        seg_sa_es = nib.load(seg_sa_es_path).get_fdata()
        seg_sa_ed = nib.load(seg_sa_ed_path).get_fdata()

        patient = {
            'sa_es': sa_es,
            'sa_ed': sa_ed,
            'seg_sa_es': seg_sa_es,
            'seg_sa_ed': seg_sa_ed
        }

        return patient
