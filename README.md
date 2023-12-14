# Sentiment Analysis For Customer Interactions In X (Formerly Twitter)
![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)

## Table of Contents
1. [Introduction](#introduction)
    - [Project Overview](#project-overview)
2. [Data Wrangling](#data-wrangling)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
5. [Modeling](#modeling)
    - [Model Evaluation Metrics](#model-evaluation-metrics)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Model Selection and Interpretation](#model-selection-and-interpretation)
6. [Conclusions and Business Recommendations](#conclusions-and-business-recommendations)
7. [References](#references)
8. [Installation](#installation)
9. [Technologies Used](#technologies-used)
10. [Contact](#contact)

## Introduction
This project utilizes machine learning to analyze and classify customer support interactions on X (formerly known as Twitter) into distinct sentiment categories. This will enable businesses to glean insights into customer sentiment, improving response strategies and overall customer satisfaction.

### Project Overview
The goal is to process and analyze textual customer support data to develop an accurate and reliable sentiment classification model. This model will help predict the sentiment of customer interactions, offering valuable feedback to the customer support teams.

## Data Wrangling
Data wrangling involved setting up a PostgreSQL database, handling missing values, and text normalization. The cleaned data was then prepared for EDA and feature extraction.
- [Data Wrangling and EDA Notebook](/notebooks/data-wrangling-cleaning.ipynb)

## Exploratory Data Analysis
We conducted an in-depth EDA to uncover underlying patterns, analyze sentiment distribution, and prepare for subsequent preprocessing and modeling stages. Insights were drawn using visualizations like word clouds and sentiment over time plots.
- [Data Wrangling and EDA Notebook](/notebooks/data-wrangling-cleaning.ipynb)

## Data Preprocessing and Feature Engineering
In this phase, we tackled issues such as text normalization, lemmatization, and feature extraction—enhancing the dataset for robust modeling.
- [Preprocessing and Modeling Notebook](notebooks/text-preprocessing-modeling.ipynb)

## Modeling
We explored several models, evaluated them on a variety of metrics, fine-tuned hyperparameters, and finally selected the most suitable model based on performance (F1 Score) and interoperability.
- [Preprocessing and Modeling Notebook](notebooks/text-preprocessing-modeling.ipynb)

### Model Evaluation Metrics
The models were evaluated using accuracy, precision, recall, and F1-score to ensure a comprehensive assessment of their performance. F1-score was chosen as the baseline metric for model selection.

### Hyperparameter Tuning
We utilized RandomizedSearchCV for an efficient search through the hyperparameter space to improve model performance.

### Model Selection and Interpretation
We selected the best-performing model and interpreted its feature importance to understand the factors influencing sentiment classification.
- [Preprocessing and Modeling Notebook](notebooks/text-preprocessing-modeling.ipynb)

## Conclusions and Business Recommendations
Our findings provide actionable insights into customer sentiment trends. We offer recommendations for leveraging this model to enhance customer service and inform business strategies.

## References
Customer Support in Twitter's dataset retrieved from Kaggle. 

## Installation
Instructions for setting up the project environment and installing dependencies.

```bash
pip install -r requirements.txt
```
## Technologies Used
Python
Pandas
scikit-learn
Matplotlib
Seaborn
XGBoost

### Contact
For inquiries or contributions, please reach out to me at camilodurangos@gmail.com.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
