# Ensemble Learning
These are files aimed to be integrated in "ptls-experiments"-like structured repository.
(https://github.com/dllllb/ptls-experiments.git)

They contain viable example of parameters which are essential for compression learning control and include some variables which should be set before the script run.

Pay attention to the necessity of yaml configs and data placed carefully for scripts functioning.

Common directory structure (you may find the appropriate directories in the [ptls-experiments repository](https://github.com/dllllb/ptls-experiments)):

`SCENARIO_ROOT`:
- `bin` - scripts for configured model runs *(place "\*.sh" files here)*
- `conf` - folder with yaml configs for model and data initialisation
- `data` - folder with dataset files

During the compression run directory `composition_results` is created. In each subfolder corresponding the combination of setup and model `checkpoints` containig serialized models and training log output is made.

## Files
Script which includes an example of `fedcore_compression` modules setup. Script may be used with inner variables change.
(more info on scripts parameters and result placements you may find at `ptls/fedcore_compression/README.md`)
* `scenario_supervised.sh` for supervised tasks and finetuning 
* `scenario_unsupervised.sh` for unsupervised tasks such as transactions encoder fitting
* `coles_age_pred.ipynb` notebook file demonstrating usage of `fc_utils` functions for experiment conduction