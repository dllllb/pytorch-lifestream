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
```

# Run scenario
 We check 5 datasets as separate experiments. See `README.md` files in experiments folder:
 - [Age](experiments/scenario_age_pred/README.md)
 - [Churn](experiments/scenario_rosbank/README.md)
 - [Assess](experiments/scenario_bowl2019/README.md)
 - [Retail](experiments/scenario_x5/README.md)
 - [Small demo dataset](experiments/scenario_gender/README.md)

# Final results
```
All results are stored in `experiments/*/results` folder.
Here are the copy of them.

Unsupervised learned embeddings with LightGBM model downstream evaluations:
                         |     mean \pm std      |
    Age group (age_pred) accuracy:
        baseline         |    0.631 \pm 0.003    |
        cpc_embeddings   |    0.594 \pm 0.002    |
        mles_embeddings  |    0.638 \pm 0.007    |
        nsp_embeddings   |    0.622 \pm 0.004    |
        rtd_embeddings   |    0.632 \pm 0.002    |
        sop_embeddings   |    0.493 \pm 0.002    |
    
    Churn (rosbank) auroc:
        baseline         |    0.825 \pm 0.004    |
        cpc_embeddings   |    0.802 \pm 0.003    |
        mles_embeddings  |    0.843 \pm 0.003    |
        nsp_embeddings   |    0.830 \pm 0.004    |
        rtd_embeddings   |    0.801 \pm 0.004    |
        sop_embeddings   |    0.782 \pm 0.005    |
        
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
    

Supervised finetuned encoder with MLP head evaluation:
                         |     mean \pm std      |
    Age group (age_pred) accuracy:
        cpc_finetuning   |    0.615 \pm 0.009    |
        mles_finetuning  |    0.644 \pm 0.004    |
        rtd_finetuning   |    0.635 \pm 0.006    |
        target_scores    |    0.628 \pm 0.004    |
    
    Churn (rosbank) auroc:
        cpc_finetuning   |    0.810 \pm 0.006    |
        mles_finetuning  |    0.827 \pm 0.004    |
        nsp_finetuning   |    0.818 \pm 0.006    |
        rtd_finetuning   |    0.819 \pm 0.005    |
        target_scores    |    0.817 \pm 0.009    |
        
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
