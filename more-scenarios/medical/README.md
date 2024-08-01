# Evaluating Fairness for Semi-supervised Cardiac Magnetic Resonance Image Segmentation

## Results

- [Pipeline analysis](./tasks_analysis.ipynb)
- [Results of Experiment 1](./ukbb_exp1_analysis.ipynb)
- [Results of Experiments 2-4](./ukbb_repeated_exps_analysis.ipynb)
- [Train and Test Data Analysis](./ukbb_sets_analysis.ipynb)

## Segmentation Models

- [UNet](./model/unet.py)
- [UNet Multi Modal](./model/unet_multi_modal.py)
- [UNet Multi Task](./model/unet_multi_task.py)

## Setup

```bash
cd UniMatch
conda create -n unimatch python=3.10.4
conda activate unimatch
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

## Inference

```bash
sbatch slurm/ukbb/unimatch/test.sh
```

or to run inference 10 times on the 10 sampled sets for a certain experiment but adjust main function arguments:
```bash
python repeat_exps.py
```

you can choose which test set to run `test.yml`:
```sh
control='ethn' # test set ['ethn', 'sex']
```

`config.yml`:
```yml
visualize: False # whether or not it prediction images should be saved
task: seg_only_mid_slices # options: ['seg_only', 'multi_task', 'multi_modal', 'seg_only_mid_slices']
seg_model: 'unet' # options: ['unet', 'unet_multi_modal', 'unet_multi_task']
```

## Training

### UKBB Dataset Train and Test Sets generation

this will generate 4 train experiments and 2 test sets balanced according to different protected attributes. Each training experiment is sampled 10 times which makes each experiment contain 10 train sets.

```bash
python gen_splits.py
```

### Training

train repeated experiments:
```bash
python repeat_exps.py
```

or for running one exp with certain setup:
```bash
sbatch slurm/ukbb/unimatch/train.sh
```

Please modify your dataset path in configuration files.

## ACDC Dataset

- ACDC: [image and mask](https://drive.google.com/file/d/1LOust-JfKTDsFnvaidFOcAVhkZrgW1Ac/view?usp=sharing)

Please modify your dataset path in configuration files.

```
├── [Your ACDC Path]
    └── data
```

to download

```
gdown 1LOust-JfKTDsFnvaidFOcAVhkZrgW1Ac
unzip ACDC.zip
```

# UniMatch for Medical Image Segmentation

We provide the official PyTorch implementation of our UniMatch in the scenario of **semi-supervised medical image segmentation**:

> **[Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2208.09910)**</br>
> [Lihe Yang](https://liheyoung.github.io), [Lei Qi](http://palm.seu.edu.cn/qilei), [Litong Feng](https://scholar.google.com/citations?user=PnNAAasAAAAJ&hl=en), [Wayne Zhang](http://www.statfe.com), [Yinghuan Shi](https://cs.nju.edu.cn/shiyh/index.htm)</br>
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2023*


## Results

**You can refer to our [training logs](https://github.com/LiheYoung/UniMatch/blob/main/more-scenarios/medical/training-logs) for convenient comparisons during reproducing.**

### ACDC


| Method                      | 1 case        | 3 cases       | 7 cases       |
| :-------------------------: | :-------: | :-------: | :-------: |
| SupBaseline                 | 28.5      | 41.5      | 62.5      |
| [UA-MT](https://arxiv.org/abs/1907.07034)             | N/A      | 61.0      | 81.5      |
| [CPS](https://arxiv.org/abs/2106.01226)                        | N/A      | 60.3      | 83.3      |
| [CNN & Transformer](https://arxiv.org/abs/2112.04894)                       | N/A      | 65.6      | 86.4      |
| **UniMatch (Ours)**         | **85.4**  | **88.9**  | **89.9**  |




## Usage

### UniMatch

```bash
# use torch.distributed.launch
# switch to current folder
sh scripts/train.sh <num_gpu> <port>
```

or
```bash
sbatch scripts/slurm_train.sh
```

To train on other datasets or splits, please modify
``dataset`` and ``split`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/more-scenarios/medical/scripts/train.sh).


### Supervised Baseline

Modify the ``method`` from ``'unimatch'`` to ``'supervised'`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/more-scenarios/medical/scripts/train.sh), and double the ``batch_size`` in configuration file if you use the same number of GPUs as semi-supervised setting (no need to change ``lr``). 


## Labels

### 7 Labels

For the case of 10% labeled (7 cases out of 70)

Patient 21: HCM
Patient 38: HCM
Patient 49: MINF
Patient 50: MINF
Patient 58: MINF
Patient 99: RV
Patient 100: RV

## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{unimatch,
  title={Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation},
  author={Yang, Lihe and Qi, Lei and Feng, Litong and Zhang, Wayne and Shi, Yinghuan},
  booktitle={CVPR},
  year={2023}
}
```


## Acknowledgement

The processed ACDC dataset is borrowed from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).
