# Composite Compression Learning

This package is aimed to widen the opportunities of standard PTLS models training by introducing whole new branch of algorithms for:
* size reduction
* latency decrease
* throughput increase

All the algorithms implementing PEFT techniques (such as *prining*, *low rank desomposition*, *quantization*, and *distillation* are encapsuleted in [FedCore library ](https://github.com/v1docq/FedCore)) and here you may find some important adaptor functions to simplify interaction.

# Useful modules
Two following modules are the essential ones for custom training process:
## fc_utils.py
* *fedcore_fit* - implements all the compression logic for specified model, data module, experimental setup and loss

* *extract_loss* - simplifies loss transportation from ptls model or given config file to the FedCore's trainer.

* *eval_computational_metrics* - allows to estimate latency, throughput ans model's size. (If called separately requires *torch.utils.data.DataLoader* instead of *PTLSDataModule*)

* *get_experimental_setup* - downloads or chooses main settings and a compression pipeline. 

## fc_setups.py
Currently, the following setups are available:

- RAW: basic setup needed for uncopmressed learning. Main purpose is to get time required for classic-style fitting

- TEST: specialised setup to fast check the large model compatibility with  all the approaches included in framework. In case some errors raise: you are welcome to create an issue at [FedCore repository ](https://github.com/v1docq/FedCore) and we will do our best to solve it.

- COMPOSITE_1: 
    [*training - prining - low rank - pruning*] is a scenario which is more appropriate for getting more interpretable weights, but a bit more slower.

- COMPOSITE_2: 
    [*training_model - low_rank_model - pruning_model - low_rank_model*]
    reaches more significant compression with respect to parameters number with more dense weight matrices. However, is often results in higher final losses.
- QAT_1: 
    composite compression including quantization aware training.
- PTQ_1: 
    composite compression including post training quantization.

## fc_fit_target.py
Module for supervised learning and finetuning. The essential options to control and include in config file are:
- pretrained - str - path to .p/./pt/.pth file with pretrained encoder. If found, the training phase may be skipped.
- setup - str - path to yaml config or name specified in *fc_setups.py* defining the compression approach.
- model_name - str - model identifier
- limit_train_batches - int - upper limit for batches (useful for overfitting avoidance or experimant acceleration)
- limit_valid_batches - int - same, but for validation and calibration.
- need_fedot_pretrain - bool - if true, fedot pretrain is used for further evolutional optimization (experimental)
- need_evo_opt - bool - if true, the 'initial assumption' form experimental setup is used as start point for evolutionary optimization 
- distributed_compression - bool - either to use Dask distributed computations or not.

All results are saved into the folder 'composition_results/{setup_name}{model_name if given}'.

Embeddings validation path may be specified with embedding_validation_results.output_path

## fc_inference.py
- inference_model - str - path to a saved model or checkpoints directory
- inference.output - str - path to a directory embeddings will be placed in.
- n_batches_computational - int - batches number for preformance evaluation. If greater than 0, runs the computational metrics estimation
- limit_predict_batches - int 
- limit_test_batches - int 
 - limit_val_batches - int

## fc_train.py
- save_encoder - str - if added, save sequence encored at the end of training.

The next ones copy those from *fc_fit_target.py*
- pretrained - str - path to .p/./pt/.pth file with pretrained encoder. If found, the training phase may be skipped.
- setup - str - path to yaml config or name specified in *fc_setups.py* defining the compression approach.
- model_name - str - model identifier
- limit_train_batches - int - upper limit for batches (useful for overfitting avoidance or experimant acceleration)
- limit_valid_batches - int - same, but for validation and calibration.
- need_fedot_pretrain - bool - if true, fedot pretrain is used for further evolutional optimization (experimental)
- need_evo_opt - bool - if true, the 'initial assumption' form experimental setup is used as start point for evolutionary optimization 
- distributed_compression - bool - either to use Dask distributed computations or not.
