# YelpRecommendation

## Introduction
This project focuses on matching benchmark performance in recommendation systems using the Yelp 2018 dataset. The dataset includes detailed reviews, user profiles, and business metadata, which are crucial for personalized recommendation systems. 

### Data Overview
- **Reviews**: Text reviews and ratings from users for various businesses.
- **Users**: Demographic and preference information of users.
- **Businesses**: Attributes of businesses including location, category, and operational hours.

### Models Implemented
- **Collaborative Filtering**: Predicts user preferences based on user-item interactions.
  - **[Collaborative Denoising Auto-Encoders (2016)](https://alicezheng.org/papers/wsdm16-cdae.pdf)** applies Denoising Auto-Encoders (DAE) to top-N recommendation systems, generalizing various collaborative filtering (CF) models. Unlike AutoRec from 2015, CDAE incorporates a user node and uses corrupted input preferences.
- **Matrix Factorization**: Reduces the dimensionality of the interaction matrix to uncover latent features.
- **Deep Neural Networks**: Leverages deep learning to enhance prediction accuracy using complex feature interactions.
- **Hybrid Models**: Integrates several models to capitalize on their individual strengths for superior performance.

Our goal is to provide a robust analysis of these models and evaluate their performance comprehensively.

## Project Structure
```
.
├── README.md
├── __init__.py
├── configs
│   ├── cdae_sweep_config.yaml
│   ├── data_preprocess.yaml
│   ├── mf_sweep_config.yaml
│   ├── sweep_config.yaml
│   └── train_config.yaml
├── data
│   ├── __init__.py
│   ├── data_preprocess.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── cdae_data_pipeline.py
│   │   ├── cdae_dataset.py
│   │   ├── data_pipeline.py
│   │   ├── dcn_data_pipeline.py
│   │   ├── dcn_dataset.py
│   │   ├── mf_data_pipeline.py
│   │   ├── mf_dataset.py
│   │   ├── ngcf_data_pipeline.py
│   │   ├── ngcf_dataset.py
│   │   ├── poprec_data_pipeline.py
│   │   ├── poprec_dataset.py
│   │   ├── s3rec_data_pipeline.py
│   │   └── s3rec_dataset.py
├── loss.py
├── metric.py
├── models
│   ├── base_model.py
│   ├── cdae.py
│   ├── dcn.py
│   ├── mf.py
│   ├── ngcf.py
│   ├── s3rec.py
│   └── wdn.py
├── poetry.lock
├── pyproject.toml
├── train.py
├── trainers
│   ├── __init__.py
│   ├── base_trainer.py
│   ├── cdae_trainer.py
│   ├── dcn_trainer.py
│   ├── mf_trainer.py
│   ├── ngcf_trainer.py
│   ├── poprec_trainer.py
│   └── s3rec_trainer.py
└── utils.py
```

## Development Environment
To run this project, you will need:
- **Python 3.11+**: Ensure Python version is up to date for compatibility.
- **Jupyter Notebook**: For interactive data analysis and visualizations.
- **Required Libraries**: pandas, numpy, scikit-learn, tensorflow/pytorch (depending on model choice).
- **Operating System**: Compatible with Windows, macOS, and Linux.

## Technology Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Vim](https://img.shields.io/badge/VIM-%2311AB00.svg?style=for-the-badge&logo=vim&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)

## Model Performance Comparison

The following table shows the performance of different models used in the project. Each model was evaluated based on multiple metrics:

| Model                   | MAP@10 | Precision@10 | Recall@10 | NDCG@10 | HIT@10 | MRR |
|-------------------------|----------|-----------|--------|----------|---------|-------|
| CDAE                    | 0.02222    | 0.01538     | 0.0713  | 0.02198    | - | - |
| DCN                     | 0.0004    | 0.0004     | 0.0016  | 0.0005    | - | - |
| NGCF                    | 0.0002    | 0.0001     | 0.0006  | 0.0002    | - | - |
| S3Rec                   | -    |   -   | -  | 0.1743  | 0.3134 | 0.1537 |
| Multi-armed bandit      | -    | -     | -  | -   | - | - |

These results were obtained from the Yelp 2018 dataset under controlled test conditions.

## How to Run

Prerequisites
- Python >= 3.11
- Poetry >= 1.8.2
- [Pytorch](https://pytorch.org/)

```
# set environments
$ poetry install
$ poetry shell

# generate input data
# download data from [yelp official website](https://www.yelp.com/dataset/download) and set data directory in config
$ vi configs/data_preprocess.yaml
$ python data/data_preprocess.py

# train model
$ vi configs/train_config.yaml
$ python train.py
```

## Contributors
<table align="center">
  <tr height="205px">
    <td align="center" width="200px">
      <a href="https://github.com/twndus"><img src="https://github.com/twndus.png" width="150px;" alt=""/></a>
    </td>
    <td align="center" width="200px">
      <a href="https://github.com/GangBean"><img src="https://github.com/GangBean.png" width="150px;" alt=""/></a>
    </td>
  </tr>
  <tr>
    <td align="center" width="200px">
      <a href="https://github.com/twndus">이주연</a>
    </td>
    <td align="center" width="200px">
      <a href="https://github.com/GangBean">조성홍</a>
    </td>
  </tr>
</table>
