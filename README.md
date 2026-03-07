# Fraudulent Credit Card Detection

The purpose of this Machine Learning application is to clasify a set of Credit Card Transactions between `Normal` and `Fraudulent` on a highly imbalanced dataset ($0.17%$ `Fraudulent` transactions) by using unsupervised models designed for Outlier Detection.

## Application Structure
1. Data Analysys and Preprocessing:
    - highly imbalanced dataset with no missing data;
    - preprocessing the data by using the *Z-Score* (`StandardScaler`);

2. Building and training the Outlier Detection Models:
    - *K-Nearest Neighbors* (KNN) ($k = 20$): classifies based on the $k$ nearest neighbors, based on the Euclidean distance;
    - *One-Class Suport Vector Machine* (OCSVM): determines the smallest hypersphere that excludes the outliers;
    - *Angle-Based Outlier Detection* (ABOD): a stochastic method that tries to maximize the cosine angle between a normal data and an outlier.

3. Evaluating the performance of each model:
    - Evaluation Scores used: *Dunn Distance*, *Davies-Bouldin Score*;
    - Based on both scores, the OCSVM is the weakest model for this dataset, while the KNN is marginaly the best of the 3 models.

4. Visualising the data by using the *PCA* reduction to plot the data in 2 dimensions. 

## Setup for the application
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data).
2. Create a virtual enviroment for the Jupyter Notebook and add the `creditcard.csv` and `Detection.ipynb` in the virtual enviroment `venv` :
``` bash
python -m venv ./venv
mv creditcard.csv Detection.ipynb ./venv
```
3. Install the Python libraries (specified in the `Technology Used` section)

## Technology Used
1. `pandas`
2. `numpy`
3. `matplotlib` and `seaborn`, for data visualization
4. Scikit-Learning (`sklearn`) for *PCA* and evaluation metrics
5. `pyod` (Python Outlier Detection) for the algorithms
