# Portfolio
---

## Life Science 
### Time-series Clustering Analysis

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/indrikwijaya/FYP-ML-For-Genomics)
[![Final Presentation Poster](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/final_presentation.pdf)
[![Full Thesis](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/IndrikWijaya_FYP_final.pdf)

<div style="text-align: justify"> 
  In this project, I evaluated different clustering algorithms to draw meaningful patterns from short time-series genomics data. Popular algorithms such as K-Means, as well as specially-developed algorithm for short time-series such as [STEM](https://link.springer.com/article/10.1186/1471-2105-7-191) are analyzed. I observed that classical clustering algorithms still performed well for various distance measure except for euclidean distance. In addition, STEM excludes many relevant genes. Thus, this study concludes that we can use STEM to get optimal number of clusters and then use any of the classical clustering algorithms to cluster our time-series data.
</div>

---

### Autoencoder for Integration of Multi-omics Data
[![View in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)]
(https://colab.research.google.com/drive/15vrcuS_L48_YeixSK1Kao0g8qAXKU-wU?usp=sharing)
<div style="text-align: justify"> 
  Multi-omics data have recently gained popularity with the advancement of sequencing and -omics technologies. Each -omic data represents an important layer in solving biological problem. By integrating different -omics data, we are able to gain novel information and useful insights that are not present when we only look at individual omic layer. Autoencoder is performed here since it's been successful in finding accurate lower dimensional representation for many studies. </div>
  
---

### Graph Convolutional Network for Protein Interaction
---

### Spatio-temporal Analysis of Translational Regulation in Brown Fat Differentiation
Following up on my thesis above, we would like to discover any further temporal and spatial patterns of variation in our data. Specifically, we want to understand how genes are translationally regulated differently at the bulk level, cytosolic level and crude mitochondria level. Here, we use [MEFISTO](https://www.nature.com/articles/s41592-021-01343-9) (developed based on [MOFA](https://biofam.github.io/MOFA2/)) which is a factor analysis model that has effectively and widely used for multi-modal genomics data sets in an unsupervised manner. 
---

## Drug Discovery
---

### Drug's Mechanism of Action (MOA) Prediction using TabNet, Deep Neural Network and Convolutional Neural Network
  
This [Kaggle's Competition](https://www.kaggle.com/c/lish-moa) coincides nicely with my current Drug Discovery project. Here, I explored various exploratory data analysis (EDA) steps, feature engineering methods and ML algorithms from top performing Kagglers. By finding the optimal model, I hope to discover suitable drugs that can help induce autophagy especially through TFEB transcription factor.
---

### Predicting Drug's Attributes from Published Perturbation Studies and Small Molecule Features
[![View in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1E9ZkeH_UAOVz03DqvbDYN17zsBpd9ry4)
<div style="text-align: justify">I explored various published datasets in order to discover effective drug(s) or small molecule(s) for autophagy inducer, particularly through TFEB activation. This pipeline is adapted from [Drugmonizome-ML](https://appyters.maayanlab.cloud/Drugmonizome_ML/) which already contains extensive drug and small molecules databases</div>
---

### Supervised Regression models of Acetylcholinesterase Inhibitors
[![View in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1BqrzHc5YYT2NSVdtyPd0i4disLLSBqEI?usp=sharing)

<div style="text-align: justify"> This project is adapted from [Data Professor's Data Science Project](https://github.com/dataprofessor/bioinformatics_freecodecamp) where multiple regression models are evaluated to predict the activity of different acetylcholinesterase inhibitors. Data is obtained from [ChEMBL Database](https://www.ebi.ac.uk/chembl/) which contains curated bioactivity data of > 2 million compounds. ChEMBL is a commonly used resource for drug discovery projects. </div>

---
## Miscellanous

### Supervised Learning for Popular Biological Data
[![View in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1XPgzCcsGid994SZcoZtuiYEnLidgngqo?usp=sharing)
<div style="text-align: justify">This exploratory project aims to investigate how Supervised ML can be applied into various biological problems. Both Classification and Regression tasks are explored here. Classical ML algorithms such as KNN, SVM and Random Forest are evaluated for this study. There are 3 mini tasks for this project: </div>

1) Classification of Breast Cancer Data

2) Regression Diabetest Patient Data

3) Cancer Classification based on Gene Expression

---
### Credit Fraud Prediction

[![Final Report](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](pdf/credit_fraud_prediction.pdf)

<div style="text-align: justify"> In this class project, my group studied various machine learning algorithms to predict classical problem in machine learning, credit fraud. Surprisingly, decision tree classifiers that typically have higher tendency to overfit the data seem to perform better against robust ensemble methods like Random Forest or AdaBoost. We hypothesized that this happens due to the bias in the datasets. Here, we proposed some suggestions to overcome this bias, particularly on dealing with imbalance data.</div>
---


---
<center>© 2020 Khanh Tran. Powered by Jekyll and the Minimal Theme.</center>
