# Portfolio
---

## Life Science 
### Time-series Clustering Analysis

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/indrikwijaya/FYP-ML-For-Genomics)
[![Final Presentation Poster](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/final_presentation.pdf)
[![Full Thesis](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/IndrikWijaya_FYP_final.pdf)

<div style="text-align: justify"> In this project, I evaluated different clustering algorithms to draw meaningful patterns from short time-series genomics data. Popular algorithms such as K-Means, as well as specially-developed algorithm for short time-series such as [STEM](https://link.springer.com/article/10.1186/1471-2105-7-191) are analyzed. I observed that classical clustering algorithms still performed well for various distance measure except for euclidean distance. In addition, STEM excludes many relevant genes. Thus, this study concludes that we can use STEM to get optimal number of clusters and then use any of the classical clustering algorithms to cluster our time-series data.</div>

---

### Supervised Learning for Popular Biological Data
[![View in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1XPgzCcsGid994SZcoZtuiYEnLidgngqo?usp=sharing)
<div style="text-align: justify">This exploratory project aims to investigate how Supervised ML can be applied into various biological problems. Both Classification and Regression tasks are explored here. There are 3 mini tasks for this project: </div>
1) Classification of Breast Cancer Data
2) Regression Diabetest Patient Data
3) Cancer Classification based on Gene Expression

---

### Autoencoder for Integration of Multi-omics Data
---

### Graph Convolutional Network for Protein Interaction
---

## Drug Discovery
---

### Deep Neural Network for Drug's Mechanism of Action (MOA) Prediction 
---

### Supervised Learning to Predict Drug's Attributes from Published Perturbation Studies and Small Molecule Features
[![View in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://drive.google.com/file/d/1E9ZkeH_UAOVz03DqvbDYN17zsBpd9ry4/view?usp=sharing)
<div style="text-align: justify">I explored various published datasets in order to discover effective drug(s) or small molecule(s) for autophagy inducer, particularly through TFEB activation. </div>
---

### Supervised Regression models of Acetylcholinesterase Inhibitors
[![View in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1BqrzHc5YYT2NSVdtyPd0i4disLLSBqEI?usp=sharing)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/detect-food-trends-facebook.html)

<div style="text-align: justify"> This project adapted from [Data Professor's Data Science Project](https://github.com/dataprofessor/bioinformatics_freecodecamp) where multiple regression models are evaluated to predict the activity of different acetylcholinesterase inhibitors. Data is obtained from [ChEMBL Database](https://www.ebi.ac.uk/chembl/) which contains curated bioactivity data of > 2 million compounds. ChEMBL is a commonly used resource for drug discovery projects. </div>

---
## Miscellanous

### Credit Fraud Prediction

[![Final Presentation Poster](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/credit_fraud_prediction.pdf)

<div style="text-align: justify"> In this class project, my group studied various machine learning algorithms to predict classical problem in machine learning, credit fraud. Surprisingly, decision tree classifiers that typically have higher tendency to overfit the data seem to perform better against robust ensemble methods like Random Forest or AdaBoost. We hypothesized that this happens due to the bias in the datasets. Here, we proposed some suggestions to overcome this bias, particularly on dealing with imbalance data.</div>
---


### Detect Food Trends from Facebook Posts: Co-occurence Matrix, Lift and PPMI

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/detect-food-trends-facebook.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/facebook-detect-food-trends)

<div style="text-align: justify">First I build co-occurence matrices of ingredients from Facebook posts from 2011 to 2015. Then, to identify interesting and rare ingredient combinations that occur more than by chance, I calculate Lift and PPMI metrics. Lastly, I plot time-series data of identified trends to validate my findings. Interesting food trends have emerged from this analysis.</div>
<br>
<center><img src="images/fb-food-trends.png"></center>
<br>

---
### Detect Spam Messages: TF-IDF and Naive Bayes Classifier

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/detect-spam-nlp.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/detect-spam-messages-nlp/blob/master/detect-spam-nlp.ipynb)

<div style="text-align: justify">In order to predict whether a message is spam, first I vectorized text messages into a format that machine learning algorithms can understand using Bag-of-Word and TF-IDF. Then I trained a machine learning model to learn to discriminate between normal and spam messages. Finally, with the trained model, I classified unlabel messages into normal or spam.</div>
<br>
<center><img src="images/detect-spam-nlp.png"/></center>
<br>

---
## Data Science

### Credit Risk Prediction Web App

[![Open Web App](https://img.shields.io/badge/Heroku-Open_Web_App-blue?logo=Heroku)](http://credit-risk.herokuapp.com/)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/chriskhanhtran/credit-risk-prediction/blob/master/documents/Notebook.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/credit-risk-prediction)

<div style="text-align: justify">After my team preprocessed a dataset of 10K credit applications and built machine learning models to predict credit default risk, I built an interactive user interface with Streamlit and hosted the web app on Heroku server.</div>
<br>
<center><img src="images/credit-risk-webapp.png"/></center>
<br>

---
### Kaggle Competition: Predict Ames House Price using Lasso, Ridge, XGBoost and LightGBM

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/ames-house-price.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/kaggle-house-price/blob/master/ames-house-price.ipynb)

<div style="text-align: justify">I performed comprehensive EDA to understand important variables, handled missing values, outliers, performed feature engineering, and ensembled machine learning models to predict house prices. My best model had Mean Absolute Error (MAE) of 12293.919, ranking <b>95/15502</b>, approximately <b>top 0.6%</b> in the Kaggle leaderboard.</div>
<br>
<center><img src="images/ames-house-price.jpg"/></center>
<br>

---
### Predict Breast Cancer with RF, PCA and SVM using Python

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](projects/breast-cancer.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/predict-breast-cancer-with-rf-pca-svm/blob/master/breast-cancer.ipynb)

<div style="text-align: justify">In this project I am going to perform comprehensive EDA on the breast cancer dataset, then transform the data using Principal Components Analysis (PCA) and use Support Vector Machine (SVM) model to predict whether a patient has breast cancer.</div>
<br>
<center><img src="images/breast-cancer.png"/></center>
<br>

---
### Business Analytics Conference 2018: How is NYC's Government Using Money?

[![Open Research Poster](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/bac2018.pdf)

<div style="text-align: justify">In three-month research and a two-day hackathon, I led a team of four students to discover insights from 6 million records of NYC and Boston government spending data sets and won runner-up prize for the best research poster out of 18 participating colleges.</div>
<br>
<center><img src="images/bac2018.JPG"/></center>
<br>

---
## Filmed by me

[![View My Films](https://img.shields.io/badge/YouTube-View_My_Films-grey?logo=youtube&labelColor=FF0000)](https://www.youtube.com/watch?v=vfZwdEWgUPE)

<div style="text-align: justify">Besides Data Science, I also have a great passion for photography and videography. Below is a list of films I documented to retain beautiful memories of places I traveled to and amazing people I met on the way.</div>
<br>

- [Ada Von Weiss - You Regret (Winter at Niagara)](https://www.youtube.com/watch?v=-5esqvmPnHI)
- [The Weight We Carry is Love - TORONTO](https://www.youtube.com/watch?v=vfZwdEWgUPE)
- [In America - Boston 2017](https://www.youtube.com/watch?v=YdXufiebgyc)
- [In America - We Call This Place Our Home (Massachusetts)](https://www.youtube.com/watch?v=jzfcM_iO0FU)

---
<center>© 2020 Khanh Tran. Powered by Jekyll and the Minimal Theme.</center>
