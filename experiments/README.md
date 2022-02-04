# Setup and test using pipenv

```sh
# Ubuntu 18.04

sudo apt install python3.8 python3-venv
pip3 install pipenv

pipenv sync  --dev # install packages exactly as specified in Pipfile.lock
pipenv shell
pytest

# run luigi server
luigid
# check embedding validation progress at `http://localhost:8082/`

# use tensorboard for metrics exploration
tensorboard --logdir lightning_logs/ 
# check tensorboard metrics at `http://localhost:6006/`

```

# Run scenario
 We check 5 datasets as separate experiments. See `README.md` files in experiments folder:
 - [Age](scenario_age_pred/README.md)
 - [Churn](scenario_rosbank/README.md)
 - [Assess](scenario_bowl2019/README.md)
 - [Retail](scenario_x5/README.md)
 - [Scoring](scenario_alpha_battle/README.md)
 - [Small demo dataset](scenario_gender/README.md)

# Final results
```
All results are stored in `experiments/*/results` folder.
Here are the copy of them.

Unsupervised learned embeddings with LightGBM model downstream evaluations:
                         |     mean \pm std      |
    Gender auroc:
        baseline         |    0.877 \pm 0.003    |
        cpc_embeddings   |    0.850 \pm 0.004    |
        mles2_embeddings |    0.885 \pm 0.003    |
        mles_embeddings  |    0.884 \pm 0.003    |
        nsp_embeddings   |    0.857 \pm 0.003    |
        random_encoder   |    0.589 \pm 0.008    |
        rtd_embeddings   |    0.860 \pm 0.003    |
        sop_embeddings   |    0.776 \pm 0.007    |                         
        barlow_twins     |    0.858 \pm 0.002    |
                         
    Age group (age_pred) accuracy:
        baseline         |    0.629 \pm 0.006    |
        cpc_embeddings   |    0.596 \pm 0.004    |
        mles2_embeddings |    0.637 \pm 0.006    |
        mles_embeddings  |    0.640 \pm 0.004    |
        nsp_embeddings   |    0.618 \pm 0.005    |
        random_encoder   |    0.375 \pm 0.008    |
        rtd_embeddings   |    0.632 \pm 0.008    |
        sop_embeddings   |    0.533 \pm 0.005    |
        barlow_twins     |    0.624 \pm 0.002    |
    
    Churn (rosbank) auroc:
        baseline         |    0.825  \pm 0.005   |
        cpc_embeddings   |    0.798  \pm 0.007   |
        mles2_embeddings |    0.843  \pm 0.007   |
        mles_embeddings  |    0.846  \pm 0.005   |
        nsp_embeddings   |    0.837  \pm 0.003   |
        random_encoder   |    0.724  \pm 0.009   |
        rtd_embeddings   |    0.807  \pm 0.003   |
        sop_embeddings   |    0.781  \pm 0.010   |
        barlow_twins     |    0.835  \pm 0.004   |
        
    Assessment (bowl2019) accuracy:
        baseline         |    0.602 \pm 0.005    |    
        cpc_embeddings   |    0.588 \pm 0.002    |    
        mles_embeddings  |    0.601 \pm 0.002    |    
        nsp_embeddings   |    0.581 \pm 0.003    |    
        rtd_embeddings   |    0.580 \pm 0.003    |    
        sop_embeddings   |    0.577 \pm 0.002    |    
    
    Retail (x5) accuracy:
        baseline         |    0.547 \pm 0.001    |
        cpc_embeddings   |    0.525 \pm 0.001    |
        mles_embeddings  |    0.539 \pm 0.001    |
        nsp_embeddings   |    0.425 \pm 0.002    |
        rtd_embeddings   |    0.520 \pm 0.001    |
        sop_embeddings   |    0.428 \pm 0.001    |
    
    Scoring (alpha battle) auroc:
        baseline         |    0.7792 \pm 0.0006  |
        random_encoder   |    0.6456 \pm 0.0009  |
        barlow_twins     |    0.7878 \pm 0.0009  |
        cpc              |    0.7919 \pm 0.0004  |
        mles             |    0.7921 \pm 0.0003  |
        nsp              |    0.7655 \pm 0.0006  |
        rtd              |    0.7910 \pm 0.0006  |
        sop              |    0.7238 \pm 0.0010  |


Supervised finetuned encoder with MLP head evaluation:
                         |     mean \pm std      |
    Gender auroc:
        barlow_twins     |    0.853 \pm 0.003    |
        cpc_finetuning   |    0.865 \pm 0.002    |
        mles_finetuning  |    0.871 \pm 0.003    |
        rtd_finetuning   |    0.869 \pm 0.003    |
        target_scores    |    0.858 \pm 0.005    |

    Age group (age_pred) accuracy:
        barlow_twins     |    0.616 \pm 0.004    |
        cpc_finetuning   |    0.619 \pm 0.005    |
        mles_finetuning  |    0.618 \pm 0.008    |
        rtd_finetuning   |    0.595 \pm 0.009    |
        target_scores    |    0.621 \pm 0.008    |
    
    Churn (rosbank) auroc:
        barlow_twins     |    0.822 \pm 0.002    |
        cpc_finetuning   |    0.821 \pm 0.004    |
        mles_finetuning  |    0.829 \pm 0.008    |
        nsp_finetuning   |    0.818 \pm 0.002    |
        rtd_finetuning   |    0.803 \pm 0.008    |
        target_scores    |    0.820 \pm 0.005    |
        
    Assessment (bowl2019) accuracy:
        cpc_finetuning   |    0.606 \pm 0.004    |    
        mles_finetuning  |    0.615 \pm 0.003    |    
        rtd_finetuning   |    0.586 \pm 0.003    |    
        target_scores    |    0.602 \pm 0.005    |    
    
    Retail (x5) accuracy:
        cpc_finetuning   |    0.549 \pm 0.001    |
        mles_finetuning  |    0.552 \pm 0.001    |
        rtd_finetuning   |    0.544 \pm 0.002    |
        target_scores    |    0.542 \pm 0.001    |

```
