## DIRECT-EFFECT RISK MINIMIZATION
# stage1
```sh
python -m domainbed.scripts.sweep launch\
       --algorithms=DCRM\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher
```
# select the best model trained in stage 1
```sh
python listresult --in_dir=/my/sweep/output/path\
                  --out_dir=stage2/weights
``
# stage2
```sh
python -m domainbed.scripts.sweep launch\
       --algorithms=ERM\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher
```
