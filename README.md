# DIRECT-EFFECT RISK MINIMIZATION

This repository is the implementation of [Direct-Effect Risk Minimization for Domain Generalization](URL). DRM consists of two stages: learning an indirect-effect representation in *stage 1* and removing the indirect effects in *stage 2*.

![Two-stage approach](./fig/stage12.png)

## Main Results

<div align=center><img src="./fig/radar_chart.png" alt="Results" width="400" height="400"/>

Test accuracy of o.o.d. algorithms on 5 correlation-shifted datasets and the DomainBed benchmark (avg). The pink region represents the performance of our method, while the light blue region represents the previously best-known results (implemented by [DomainBed](https://github.com/facebookresearch/DomainBed/) using training-domain validation) on each dataset.

![Results](./fig/R1.png)

![Results](./fig/R2.png)

Min and Avg are the minimum value and the average of accuracy for all test environments, respectively.

## Available datasets

The [currently available datasets](https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py) are:

- RotatedMNIST ([Ghifary et al., 2015](https://arxiv.org/abs/1508.07680))
- ColoredMNIST ([Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893))
- VLCS ([Fang et al., 2013](https://openaccess.thecvf.com/content_iccv_2013/papers/Fang_Unbiased_Metric_Learning_2013_ICCV_paper.pdf))
- PACS ([Li et al., 2017](https://arxiv.org/abs/1710.03077))
- Office-Home ([Venkateswara et al., 2017](https://arxiv.org/abs/1706.07522))
- A TerraIncognita ([Beery et al., 2018](https://arxiv.org/abs/1807.04975)) subset
- DomainNet ([Peng et al., 2019](http://ai.bu.edu/M3SDA/))
- CelebA ([Liu et al., 2019](https://openaccess.thecvf.com/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html))
- 3dshapes([Burgess et al., 2018]([Page not found · GitHub](https://github.com/deepmind/3dshapes-dataset/)))
- DSprites([Matthey et al., 2017]([deepmind/dsprites-dataset: Dataset to assess the disentanglement properties of unsupervised learning methods (github.com)](https://github.com/deepmind/dsprites-dataset/)))



## Training

The code structure of [ ./stage1](./stage1) and [./stage2 ](./stage2 )is similar to [DomainBed](https://github.com/facebookresearch/DomainBed/). Script [./stage1(2)/domainbed/scripts/train.py](./stage1/domainbed/scripts/train.py) trains a single model. Script [./stage1(2)/domainbed/scripts/sweep.py](./stage1/domainbed/scripts/sweep.py) searches hyperparameters.  Script [./stage1listresult.py](./stage1listresult.py) is used to select the best model trained in stage 1.

### stage1

```sh
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher=MyLauncher\
       --datasets=XX
```

The parameter 'datasets' can be ColoredMNIST, CelebA, DSprites, dshapes, RotatedMNIST, VLCS, PACS, OfficeHome, TerraIncognita, and DomainNet, parameter 'output_dir' is the path to save the models and results.

### select the best model trained in stage 1

```sh
python listresult --in_dir=/my/sweep/output/path\
                  --out_dir=stage2/weights
```

The parameter 'in_dir' should be consistent with 'output_dir' in stage1 parameter '--out_dir' is the path to copy the best model.

### stage2

```sh
python -m domainbed.scripts.sweep launch\
       --algorithms=ERM\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher
       --datasets=XX
```

DRM can be used in conjunction with different methods and now supports ERM, IRM, and CORAL.
