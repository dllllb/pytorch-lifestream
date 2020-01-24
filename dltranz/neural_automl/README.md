# Neural Oblivious Decision Ensembles
A supplementary code for [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312) paper.

# What does it do?
It learns deep ensembles of oblivious differentiable decision trees on tabular data

# How to run?
There are 2 modes - **Scoring mode** and with **TrxEncoder mode**
## Scoring mode
In scoring mode it works like automl. You need to run:
```
from neaural_automl_tools import train_from_config
train_from_config(X_train, y_train, X_valid, y_valid, model_config)
``` 
Here you need specify data arrays and model config. Currently data are numpy arrays (X_train - [n_samples, n_feautures], y_train - [n_samples])

If model_config is None then **neural_automl/conf/binary_classification_config.json** is used

## Config file structure
    "device": "cuda",
    
    # perform data preprocessing
    "data_aware_initialisation": 1000                # number of train examples to take for initialisation (default 10000)
    "data_transform_params": {
        "normalize": true,                           # can give comparable results as quantile_transform but faster
        "quantile_transform": false,                 # possible can get higher scores but very time consuming
        "quantile_noise": 1e-3
    },

    "train_params": {
        "batch_size": 1024,
        "n_epoch": 2
    },

    "valid_params": {
        "batch_size": 256
    },

    "sgd_params": {
        "lr": 0.004,                                 # Adam optimizer is used
        "step_size": 1,                              # Step Scheduler is used
        "step_gamma": 0.5
    },

    "loss_params": {                     
        "func": "binary_cross_entropy_with_logits"   # currently binari/multiclass/ classification and regression are supported is supported
    },

    "accuracy_params": {
        "func": "roc_auc/accuracy/mse"
    },

    "layers": [ # model architecture
        {
            "type": "batchnorm",                     # if you want you can add this layer  
            "d_in": n                                # input dimentions (default None). If not set it could be calculated from previous layers 
        },
        {
            "type": "tree",                          
            "d_in": n,                               # input dimentions (default None). If not set it could be calculated from previous layers 
            "num_trees": 128,                        # number of trees in layer
            "num_sub_layers": 2,                     # each layer behaves like Res_Net, you can create k layers from num_trees trees with residual connections among all of them 
            "tree_dim": 3,                           # usaully each tree returns scalar output, here it returns vector with size = tree_dim
            "tree_depth": 6,                         # each simmetric binary tree with depth = t, subdivide space on 2^t regions
            "flatten_output": false                  # d_out = num_trees * num_sub_layers * tree_dim if flatten_output == True else [num_trees * num_sub_layers, tree_dim], (default True)
        },
        {
            "type": "aggregation",
            "func": "mean",                          # can be mean or sum
            "dims": (i1, .., ik)                     # dimmentions along which perform operation (default = -1)
        },
        {
            "type": "linear",    
            "d_in": n,                               # input dimentions (default None). If not set it could be calculated from previous layers  
            "d_out": 1                               # output dimention
        },
        {
           "type": "sigmoid",                        # can be mean or sigmoid/relu/softmax/log_softmax
        },
        ...
        ...
        ...
    ]

"loss_params" could be one of: **cross_entropy/nll** - for multiclass classification or **binary_cross_entropy_with_logits/binary_cross_entropy** for binary classification and **mse** for regression

You also can specify **model_config** as file_name like this: **model_config='binary_classification_config.json'** which should be in **conf** subfolder

There are several configs in **conf** for binaty/multy class classification and regression
 
If you don't want you can not set input dimention d_in for each layer and then it will be calculated automaticaly in simple cases.

For example in model above, we have next input/output sizes:

```
bathnorm layer:
   d_in = from input data
   d_out = d_in
tree layer:
   d_in = batchnorm layer d_out
   d_out = (num_trees * num_sub_layers, tree_dim) = (batch, 256, 3)
   
layer aggregation:
   d_in = (batch, 256, 3)
   d_out = (bathc, 256)
   
layer linear:
   d_in = (batch, 256)
   d_out = (batch, 1)
   
layer sigmoid:
   d_in = (batch, 1)
   d_out = (batch, 1)
   
layer last: # this layer removes dimentions with sizes = 1 
   d_in = (batch, 1)
   d_out = (batch)
``` 

You can try combine this layers as possible

## Results 
for model: **common_models/model_base.p   |  common_models/ml_base.json**

Task                            | NODE ROC_AUC       | XGB ROC_AUC| LINEAR ROC_AUC |
--------------------------------| ------------------ | ---------- | ---------------|
**Shishorin st_default**        | **0.833**          | 0.832      |                |
**Shishorin mt_default**        | **0.795**          | 0.792      |                |
**Shishorin target_default**    | **0.810**          | 0.806      |                |
**Telemed**                     | **0.856**          | 0.848      |                |
**Sokolov**                     | 0.527              | **0.542**  |                |
--------------------------------|--------------------|------------|----------------|
**gender ml embeddings**        | **0.859**          | 0.853      | 0.858          |
**age ml embeddings**           | **0.617**          | 0.607      | **0.617**      | 

## Learn with TrxEncoder Mode

If you want to train models like 'RNN' + 'NODE as head' add key **'neural_automl'** to **'head''** in your model config file like this:

```
# no config specified, then use default config from neural_automl/conf/default_config.json
"head": {
      "neural_automl": ''
      "norm_input": true,
      "pred_all_states": false,
      "pred_all_states_mean": false,
      "explicit_lengths": false
    },
```

```
# your own config
"head": {
      "neural_automl": {
            "layers": [
                {
                    "type": "tree",
                    "num_trees": 128,
                    "num_sub_layers": 2,
                    "tree_dim": 3,
                    "tree_depth": 6,
                    "flatten_output": false
                },
                {
                    "type": "aggregation",
                    "func": "mean"
                },
                {
                    "type": "linear",
                    "d_out": 1
                }
            ]      
      }
      "norm_input": true,
      "pred_all_states": false,
      "pred_all_states_mean": false,
      "explicit_lengths": false
    },

```

## Results
On **shishorin_pl_united_customer** ROC_AUC = 0.888

Method                             |              ROC_AUC |
-----------------------------------| -------------------- |
**base model**                     | 0.888                |
**neural_automl (first start )**   | 0.889                |
**neural_automl with batch norm**  | **0.891**            |
-----------------------------------|----------------------|
**gender base model**              | 0.861                |
**gender neural_automl**           | **0.864**            |
-----------------------------------|----------------------|
**age base model**                 | **0.631**            |
**age neural_automl**              | 0.620                |
 

