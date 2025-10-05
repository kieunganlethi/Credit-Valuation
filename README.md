# Credit-Valuation
## Credit Evaluation using Machine Learning Models
# Overview
This project aims to build and evaluate multiple machine learning models for credit risk classification, using real banking survey data. The goal is to classify customers into five debt groups (from 1 = best to 5 = worst) based on financial and demographic features. The project applies systematic data preprocessing, feature engineering, and model evaluation techniques to ensure fair and accurate predictions, even under severe class imbalance.
# Objective
To develop a multi-class credit scoring model that predicts a customer’s credit group using both traditional and advanced algorithms, while addressing data imbalance and improving model interpretability.
# Key sub-goals include:
Cleaning, transforming, and standardizing large-scale financial data.
Handling missing values, duplicates, outliers, and unbalanced class distribution.
Comparing traditional models (Logistic Regression) and non-traditional models (Random Forest, XGBoost, CatBoost, Neural Network).
Selecting the optimal model for accurate and stable credit classification.
# Methodology
1. Data Exploration & Preparation
- Data Cleaning: Standardized inconsistent date formats, corrected data types, and removed duplicates.
- Missing Values: Filled categorical missing data using mode imputation.
- Outlier Detection: Applied the Z-score method to remove extreme financial values.
- Imbalance Handling: Used SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic samples for minority classes.
- Encoding: Applied Label Encoding to categorical variables.
- Feature Engineering: Standardized continuous features, removed highly correlated variables (threshold = 0.8), and built a scikit-learn pipeline for reproducibility.
2. Model Building and Evaluation
- Implemented and compared five models using Stratified K-Fold Cross-Validation (k=5) with SMOTE applied inside each fold:*
+ Traditional model: Logistic Regression
+ Non-traditional models: Random Forest, XGBoost, CatBoost, Neural Network
- Evaluation Metrics:
+ Accuracy
+ F1 Macro
# Validation Phase
A separate validation dataset was used to test the final Random Forest model on unseen data.
Findings:
- The model maintained strong performance for majority class (Group 1) but struggled with minority risk groups.
- Indicates the need for further improvement through class-specific tuning or data augmentation.
# Tools and Technologies
- Programming Language: Python
- Libraries: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, catboost, tensorflow/keras, matplotlib, seaborn
- Techniques: Stratified K-Fold, SMOTE, Feature Engineering, Model Evaluation
