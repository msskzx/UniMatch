import os

def get_patient_ids_from_directory(directory):
    """
    read all patients ids from directory

    Arguements:
    directory -- path

    return: list of ids
    """
    return [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]


def save_patient_ids(patient_ids, output_file):
    """
    save the list of ids in split text file

    Arguements:
    patient_ids -- list of ids
    output_file -- path
    """
    with open(output_file, 'w') as file:
        for patient_id in patient_ids:
            file.write(f'{patient_id}-sa_ED\n{patient_id}-sa_ES\n')


def get_patient_ids_frames(split):
    """
    initialize patient ids and frames 

    Argument:
    split -- split file path
    """
    with open(split, 'r') as f:
        str_ids_frames = f.read().splitlines()
    
    return [x.split('-') for x in str_ids_frames]