# DIRECT-EFFECT RISK MINIMIZATION

This repository is the implementation of [Direct-Effect Risk Minimization for Domain Generalization](URL). DRM consists of two stages: learning an indirect-effect representation in *stage 1* and removing the indirect effects in *stage 2*.

![Two-stage approach](./fig/stage12.png)

## Main Results

![Results](./fig/R1.png)

![Results](./fig/R2.png)

Min and Avg are the minimum value and the average of accuracy for all test environments, respectively.

## How to Run

### stage1

```sh
python -m domainbed.scripts.sweep launch\
       --algorithms=DCRM\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher
```

### select the best model trained in stage 1

```sh
python listresult --in_dir=/my/sweep/output/path\
                  --out_dir=stage2/weights
```

### stage2

```sh
python -m domainbed.scripts.sweep launch\
       --algorithms=ERM\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher
```
