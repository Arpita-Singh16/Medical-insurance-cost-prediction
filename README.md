# Medical-insurance-cost-prediction
Medical Insurance Cost Prediction

A machine learning project that predicts an individual's medical insurance charges from personal details (age, sex, BMI, children, smoking status and region) using Linear Regression in Python.

# Problem

Insurance companies price policies based on how much a person is expected to cost. Which personal factors drive medical charges the most, and how well can we predict the charge for a new person?

# Dataset
File: insurance.csv — 1,338 rows, 7 columns, no missing values
Columns: age, sex, bmi, children, smoker, region, charges (target)

# Approach
Exploratory data analysis (EDA) — histograms, count plots, box plots (outlier check) and a correlation heatmap
Data cleaning — removed 1 duplicate row (1,337 rows left); checked for missing values and data types
Encoding — sex → is_female, smoker → is_smoker (0/1); one-hot encoded region
Feature engineering — grouped bmi into categories (Underweight, Normal, Overweight, Obese) and one-hot encoded them
Scaling — standardized age, bmi and children with StandardScaler
Feature selection — Pearson correlation and chi-square tests of each feature against charges
Model — 80/20 train–test split (random_state=42), LinearRegression from scikit-learn
Evaluation — adjusted R² on the test set

Features used in the final model: age, is_female, bmi, children, is_smoker, region_southeast, bmi_category_Obese

# Key findings
Charges vary a lot: the average is about 13,270, but the highest is about 63,770.
About 20% of the people in the data are smokers (274 of 1,337).
Smoking is by far the strongest factor. Its correlation with charges is 0.79, compared with 0.30 for age, 0.20 for BMI and 0.07 for number of children.
Region and sex have only a weak relationship with charges.
The Linear Regression model reached an adjusted R² of about 0.80 on the held-out test set.
Tools

Python · Pandas · NumPy · Matplotlib · Seaborn · SciPy · Scikit-learn · Jupyter Notebook

# Next steps
Report error metrics too (MAE, RMSE), not only R²
Use cross-validation for a more reliable score
Compare with tree-based models (Random Forest, Gradient Boosting)
Fit the scaler on the training data only, to avoid leaking information from the test set
Test an interaction between smoking and BMI
# How to run
bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
# keep insurance.csv in the same folder as the notebook
jupyter notebook ML_project1_.ipynb

# Author
Arpita Singh — GitHub · LinkedIn
