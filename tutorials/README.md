# **Learn Event Sequence Analysis with pytorch-lifestream**

Welcome to your comprehensive guide to **Event Sequence Deep Learning** analysis using the **ptls** (pytorch-lifestream) library! Whether you're a beginner or an advanced machine learning enthusiast, this structured path will guide you through the essential topics of processing event sequences. With a mix of tutorials and hands-on examples, you’ll learn to implement everything from sequence classification to unsupervised learning methods for event-based data.

Here’s a breakdown of the key topics:

---

## **Section 1: Prerequisites**
Before diving into event sequence analysis, let’s make sure you’re equipped with some basic knowledge:

### 1.1 [**PyTorch**](https://www.youtube.com/watch?v=Z_ikDlimN6A)
Learn the basics of PyTorch, the most popular deep learning framework.  
🔗 *Video Tutorial: [Link](https://www.youtube.com/watch?v=Z_ikDlimN6A)*

### 1.2 [**PyTorch-Lightning**](https://www.youtube.com/playlist?list=PLhhyoLH6IjfyL740PTuXef4TstxAK6nGP)
Make your PyTorch code cleaner and more organized using PyTorch Lightning.  
🔗 *Video Tutorial Playlist: [Link](https://www.youtube.com/playlist?list=PLhhyoLH6IjfyL740PTuXef4TstxAK6nGP)*

### 1.3 (Optional) [**Hydra**](https://hydra.cc/)
Streamline your configurations with the Hydra framework.  
🔗 *Demo Code: [Hydra CoLES Training](./notebooks/Hydra%20CoLES%20Training.ipynb)*

### 1.4 [**pandas**](https://pandas.pydata.org/)
Master data preprocessing with pandas, the go-to library for manipulating datasets.

### 1.5 (Optional) [**PySpark**](https://spark.apache.org/docs/latest/api/python/index.html)
Handle large datasets with PySpark for efficient big data preprocessing.

---

## **Section 2: Event Sequences**
Now let’s explore the core of event sequence analysis. These sections will cover both **global** and **local** problems.

### 2.1 **Event Sequence for Global Problems**
Understand global tasks such as event sequence classification.  
*TBD*

### 2.2 **Event Sequence for Local Problems**
Delve into local tasks like next event prediction.  
*TBD*

---

## **Section 3: Supervised Neural Networks**
Build supervised models for classifying event sequences.

### 3.1 **Network Types**
Explore different types of networks for event sequence processing.

#### 3.1.1 **Recurrent Neural Networks (RNNs)**
Classic networks designed to handle sequence data.  
*TBD*

#### 3.1.2 (Optional) **Convolutional Neural Networks (CNNs)**
Use CNNs for specific sequence modeling tasks.  
*TBD*

#### 3.1.3 **[Transformers](demo/supervised-sequence-to-target-transformer.ipynb)**
Leverage the power of transformers for sequence classification.  
🔗 *Demo Code: [Transformer Notebook](./notebooks/supervised-sequence-to-target-transformer.ipynb)*

### 3.2 **Problem Types**
Understand the variety of problems you can solve with sequences.

#### 3.2.1 **Global Problems**
From binary classification to multi-label regression.  
*TBD*  
🔗 *Demo Code: [Multilabel Classification](./notebooks/multilabel-classification.ipynb)*

#### 3.2.2 **Local Problems**
Next event prediction tasks using embeddings.  
🔗 *Demo Code: [Local Embeddings](./notebooks/event-sequence-local-embeddings.ipynb)*

---

## **Section 4: Unsupervised Learning**
Learn how to pretrain self-supervised models using proxy tasks.

### 4.1 (Optional) **Word2vec**
Use context-based methods for unsupervised learning.  
*TBD*

### 4.2 **MLM, RTD, GPT**
Train self-supervised models with Masked Language Model (MLM) and others.  
🔗 *Demo Code: [MLM Embeddings](./notebooks/mlm-emb.ipynb)*  
🔗 *Demo Code: [Event Sequence Local Embeddings](./notebooks/event-sequence-local-embeddings.ipynb)*

### 4.3 **NSP, SOP**
Implement sequence-based methods like Next Sentence Prediction (NSP) and Sentence Order Prediction (SOP).  
🔗 *Demo Code: [NSP/SOP Embeddings](./notebooks/notebooks/nsp-sop-emb.ipynb)*

---

## **Section 5: Contrastive Learning**
Learn about contrastive and non-contrastive learning for latent representations.

### 5.1 **CoLES**
Train contrastive learning models with CoLES.  
🔗 *Demo Code: [CoLES Embeddings](./notebooks/coles-emb.ipynb)*

### 5.2 **VICReg**
Explore VICReg for representation learning.  
*TBD*

### 5.3 **CPC**
Learn about CPC for contrastive learning.  
*TBD*

### 5.4 **MLM, TabFormer, and Others**
Self-supervised training using MLM for transaction data.  
🔗 *Demo Code: [MLM Embeddings](./notebooks/mlm-emb.ipynb)*  
🔗 *Demo Code: [TabFormer Embeddings](./notebooks/tabformer-emb.ipynb)*

---

## **Section 6: Pretrained Model Usage**
Use pretrained models for downstream tasks or fine-tune them.

### 6.1 **Downstream Model on Frozen Embeddings**
Apply frozen embeddings to new models.  
*TBD*

### 6.2 **CatBoost with Embeddings**
Train CatBoost models on embeddings.  
🔗 *Demo Code: [CatBoost with Embeddings](demo/coles-catboost.ipynb)*

### 6.3 **Model Finetuning**
Fine-tune pretrained models for better performance.  
🔗 *Demo Code: [Finetuning](./notebooks/coles-finetune.ipynb)*

---

## **Section 7: Preprocessing Options**
Learn various preprocessing techniques for event sequences.

### 7.1 **ptls-format Parquet Data Loading**
Use PySpark and Parquet for efficient data processing.  
🔗 *Demo Code: [Parquet Data Loading](./notebooks/pyspark-parquet.ipynb)*

### 7.2 **Fast Inference for Large Datasets**
Optimizing inference for large-scale datasets.  
🔗 *Demo Code: [Extended Inference](./notebooks/extended_inference.ipynb)*

---

## **Section 8: Special Feature Types**
Explore different feature types with pretrained encoders.

### 8.1 **Pretrained Encoder for Text Features**
Use pretrained encoders for text-based features.  
🔗 *Demo Code: [Pretrained Embeddings](./notebooks/coles-pretrained-embeddings.ipynb)*

### 8.2 **Multi-source Models**
Implement models that use data from multiple sources.  
🔗 *Demo Code: [Multimodal Unsupervised Learning](./notebooks/CoLES-demo-multimodal-unsupervised.ipynb)*

---

## **Section 9: Transaction Encoding Options**
Dive into encoding strategies for transactions.

### 9.1 **Basic Encoding Options**
Learn the fundamentals of transaction encoding.  
*TBD*

### 9.2 **Transaction Quantization**
Quantization techniques for transactions.  
*TBD*

### 9.3 **Transaction BPE**
Using Byte Pair Encoding (BPE) for transactions.  
*TBD*

---

Explore these topics, experiment with the demo code, and master the art of event sequence deep learning with PyTorch-Lifestream!
