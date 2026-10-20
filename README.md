# Adult Income Classification: Data Preprocessing & Feature Selection

A Machine Learning assignment focused on understanding, cleaning, preprocessing, and selecting relevant features from the Adult Census Income dataset for a binary classification problem.

## 📌 Project Overview

The objective of this project is to perform end-to-end exploratory data analysis (EDA), data preprocessing, and feature selection on the Adult Census Income dataset. The project explores different statistical and machine learning techniques to identify potentially useful features for predicting whether an individual's annual income exceeds $50,000.

The assignment starts with basic dataset understanding and gradually progresses toward preparing the data for classification models.

## 🎯 Problem Statement

**Can an individual's income category be predicted using demographic, educational, occupational, and financial attributes?**

The task is formulated as a **supervised binary classification problem**.

The target variable is `income`, which has two classes:

- `<=50K`: Annual income is less than or equal to $50,000.
- `>50K`: Annual income is greater than $50,000.

The purpose of this assignment is to prepare the dataset for machine learning by identifying data quality issues, handling categorical and numerical features appropriately, and investigating feature selection techniques.

## 📊 Dataset Description

The Adult Census Income dataset contains demographic and employment-related information about individuals. It is commonly used for classification experiments.

### Dataset Information

| Property | Description |
|---|---|
| Dataset name | Adult Census Income Dataset |
| Total observations initially loaded | 32,561 |
| Total original columns | 15 |
| Numerical columns | 6 |
| Categorical columns | 9 |
| Target variable | `income` |
| Problem type | Binary classification |
| Final observations after duplicate removal | 32,537 |
| Input features before encoding | 14 |

### Feature Description

| Feature | Description |
|---|---|
| `age` | Age of the individual |
| `workclass` | Employment sector or work classification |
| `fnlwgt` | Census final sampling weight |
| `education` | Highest education category |
| `education-num` | Numerical encoding of education level |
| `marital-status` | Marital status |
| `occupation` | Type of occupation |
| `relationship` | Relationship category in the household |
| `race` | Recorded race category |
| `sex` | Recorded sex category |
| `capital-gain` | Capital gains |
| `capital-loss` | Capital losses |
| `hours-per-week` | Hours worked per week |
| `native-country` | Country of origin |
| `income` | Income category; target variable |

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualizations
- **Scikit-learn** — Preprocessing and feature selection
- **Jupyter Notebook / Google Colab** — Interactive experimentation
- **Git and GitHub** — Version control and project documentation

## 📁 Suggested Repository Structure

```text
adult-income-classification/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── adult_income_analysis.ipynb
│
├── images/
│   ├── age_boxplot.png
│   ├── numerical_boxplots.png
│   └── correlation_heatmap.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

**Note:** The folder and file names above are suggested. Keep only the files that actually exist in your repository. The dataset itself can be downloaded separately instead of committing a large data file.

## 🔍 Project Workflow

The assignment follows these major stages:

1. Dataset loading and understanding
2. Data type identification
3. Dataset structure and statistical summary
4. Categorical value exploration
5. Missing-value investigation
6. Duplicate detection and removal
7. Target variable distribution analysis
8. Outlier detection and interpretation
9. Feature-target separation
10. Train-test split
11. Categorical encoding
12. Numerical feature scaling
13. Processed feature identification
14. Feature selection
15. Model-based evaluation

---

## 1. Dataset Loading and Understanding

The dataset was loaded into a Pandas DataFrame.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("adult.csv")

print(df.head())
print(df.shape)
print(df.info())
```

### Purpose

The first step is to understand the dataset's structure, number of observations, number of columns, and data types.

### Initial Observation

- 32,561 observations
- 15 columns
- 6 numerical columns
- 9 categorical columns

The `income` column was identified as the target variable.

## 2. Data Type Identification

The dataset contains both numerical and categorical features.

Numerical columns:

```python
numerical_features = df.select_dtypes(
    include=np.number
).columns.tolist()

print(numerical_features)
```

Categorical columns:

```python
categorical_features = df.select_dtypes(
    include="object"
).columns.tolist()

print(categorical_features)
```

### Why Is This Important?

Numerical and categorical features require different preprocessing techniques. Numerical features may require scaling, while categorical features generally require encoding before being used by most machine learning algorithms.

## 3. Dataset Information and Statistical Summary

The dataset structure was examined using:

```python
df.info()
df.describe()
df.describe(include="object")
```

### Purpose

- `df.info()` displays column data types and non-null counts.
- `df.describe()` summarizes numerical features.
- `df.describe(include="object")` summarizes categorical features, including unique values and most frequent categories.

### Observations

The numerical summary showed differences in feature ranges. For example, `age` ranged from 17 to 90, while `fnlwgt` and `capital-gain` had substantially larger numerical values.

The categorical summary showed that `Private` was the most frequent workclass, `HS-grad` was the most frequent education category, and `<=50K` was the most frequent income class.

## 4. Categorical Value Exploration

Unique categories were inspected to understand the values present in each categorical feature.

```python
for column in categorical_features:
    print(f"\n{column}:")
    print(df[column].unique())
```

### Purpose

This step helps identify category variations, inconsistent labels, and special values such as `?`, which represent unknown or missing categorical information in this dataset.

## 5. Missing-Value Investigation

The missing values were examined using:

```python
df.isnull().sum()
```

In this dataset, missing categorical values were represented by `?` rather than actual Pandas null values.

The relevant columns were:

- `workclass`: 1,836 occurrences of `?`
- `occupation`: 1,843 occurrences of `?`
- `native-country`: 583 occurrences of `?`

Their missing percentages were approximately:

| Feature | Missing percentage |
|---|---:|
| `workclass` | 5.64% |
| `occupation` | 5.66% |
| `native-country` | 1.79% |

### Important Observation

The `?` values must be handled explicitly because `df.isnull()` does not automatically recognize them as missing values.

For example, they can be converted into `NaN` for missing-value handling or replaced with a consistent label such as `Unknown`, depending on the preprocessing strategy.

## 6. Duplicate Detection

Duplicate rows were checked using:

```python
duplicate_count = df.duplicated().sum()
print("Number of duplicate rows:", duplicate_count)
```

### Result

**24 duplicate rows** were identified.

They were removed using:

```python
df = df.drop_duplicates().reset_index(drop=True)
```

After removing duplicates, the dataset contained **32,537 observations**.

### Why Remove Duplicates?

Duplicate records can cause repeated observations to receive excessive weight during model training. However, duplicates should be removed only after considering whether they represent accidental repetition or legitimate repeated observations.

## 7. Target Variable Distribution

The distribution of the target variable was examined using:

```python
print(df["income"].value_counts())
print(df["income"].value_counts(normalize=True) * 100)
```

### Observed Distribution

| Income class | Count | Percentage |
|---|---:|---:|
| `<=50K` | 24,720 | 75.92% |
| `>50K` | 7,841 | 24.08% |
| **Total** | **32,561** | **100%** |

These counts describe the distribution before duplicate removal.

### Interpretation

The target variable is imbalanced because approximately 75.92% of observations belong to the `<=50K` class, while 24.08% belong to the `>50K` class.

Therefore, model evaluation should include precision, recall, F1-score, and a confusion matrix in addition to accuracy.

## 8. Outlier Detection

Boxplots were generated for the numerical features to visualize their distributions and identify potential outliers.

```python
import matplotlib.pyplot as plt
import seaborn as sns

for column in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()
```

The Interquartile Range (IQR) method was also used to count potential outliers.

The IQR is defined as:

\[
IQR = Q_3 - Q_1
\]

The conventional bounds are:

\[
\text{Lower Bound} = Q_1 - 1.5 \times IQR
\]

\[
\text{Upper Bound} = Q_3 + 1.5 \times IQR
\]

### Observations

Potential outliers were detected in multiple numerical features, particularly `fnlwgt`, `capital-gain`, `capital-loss`, and `hours-per-week`.

These values were not automatically removed because extreme values can represent legitimate observations. Outlier treatment should depend on domain knowledge and the requirements of the downstream model.

## 9. Feature-Target Separation

The target variable was separated from the input features.

```python
X = df.drop("income", axis=1)
y = df["income"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)
```

### Result

- Input features: 14 columns
- Target variable: 1 column
- Observations after duplicate removal: 32,537

This is a supervised learning problem because the target labels are available during training.

## 10. Train-Test Split

The dataset was divided into training and testing subsets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Result

| Dataset | Observations |
|---|---:|
| Training set | 26,029 |
| Testing set | 6,508 |
| Total | 32,537 |

### Why Use `stratify=y`?

Stratified splitting approximately preserves the target class proportions in both subsets.

`random_state=42` ensures that the split is reproducible.

**Important:** The test set is kept separate from preprocessing steps that learn parameters from data, feature selection, and model fitting. Such steps must be fitted using training data only.

## 11. Categorical Encoding

One-Hot Encoding was applied to the eight categorical input features.

```python
from sklearn.preprocessing import OneHotEncoder

categorical_cols = X_train.select_dtypes(
    include="object"
).columns.tolist()

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_cat = encoder.fit_transform(
    X_train[categorical_cols]
)

X_test_cat = encoder.transform(
    X_test[categorical_cols]
)
```

### Result

| Dataset | Encoded categorical features |
|---|---:|
| Training | 102 |
| Testing | 102 |

### Why One-Hot Encoding?

One-Hot Encoding converts categorical values into binary columns. It avoids assigning arbitrary numerical rankings to nominal categories.

The encoder is fitted on the training set and then applied to the testing set to avoid data leakage.

## 12. Numerical Feature Scaling

StandardScaler was used to standardize the six numerical input features.

```python
from sklearn.preprocessing import StandardScaler

numerical_cols = X_train.select_dtypes(
    include=np.number
).columns.tolist()

scaler = StandardScaler()

X_train_num = scaler.fit_transform(
    X_train[numerical_cols]
)

X_test_num = scaler.transform(
    X_test[numerical_cols]
)
```

The transformation follows:

\[
z = \frac{x-\mu}{\sigma}
\]

Here, the mean and standard deviation are learned from the training data.

### Result

- Training numerical matrix: 26,029 × 6
- Testing numerical matrix: 6,508 × 6

Scaling is especially useful for algorithms such as Logistic Regression, Support Vector Machines, and distance-based methods.

## 13. Combining Processed Features

The scaled numerical features and one-hot encoded categorical features were combined.

```python
X_train_processed = np.hstack([
    X_train_num,
    X_train_cat
])

X_test_processed = np.hstack([
    X_test_num,
    X_test_cat
])
```

### Result

| Dataset | Rows | Processed features |
|---|---:|---:|
| Training | 26,029 | 108 |
| Testing | 6,508 | 108 |

The final processed feature matrices contain 6 scaled numerical features and 102 encoded categorical features.

---

# 🧠 Feature Selection

Feature selection identifies a subset of input features that may be useful for a prediction task. It can reduce dimensionality, training time, and model complexity.

Different techniques evaluate features in different ways. Therefore, their results should not be compared solely by the number of features selected.

## 14. Variance Threshold

Variance Threshold is a filter-based method that removes features with variance below a specified threshold.

```python
from sklearn.feature_selection import VarianceThreshold

selector_var = VarianceThreshold(threshold=0.01)

X_train_var = selector_var.fit_transform(
    X_train_processed
)

X_test_var = selector_var.transform(
    X_test_processed
)
```

### Observed Result

| Metric | Value |
|---|---:|
| Features before selection | 108 |
| Features after selection | 60 |
| Features removed | 48 |
| Dimensionality reduction | 44.44% |

### Interpretation

The threshold removed low-variance features, including some infrequent one-hot encoded categories.

Variance Threshold does not use the target variable, so retained features are not necessarily the most predictive features.

## 15. Correlation Analysis

A correlation matrix was generated for the numerical input features and a binary representation of the target.

```python
correlation_data = X_train[numerical_cols].copy()

correlation_data["income"] = y_train.map({
    "<=50K": 0,
    ">50K": 1
})

corr_matrix = correlation_data.corr()
```

### Observed Correlations with Income

| Feature | Correlation |
|---|---:|
| `education-num` | 0.34 |
| `age` | 0.23 |
| `hours-per-week` | 0.23 |
| `capital-gain` | 0.22 |
| `capital-loss` | 0.15 |
| `fnlwgt` | -0.01 |

### Interpretation

`education-num` showed the strongest positive linear correlation with income among the numerical features.

No strong pairwise linear correlations were observed among the numerical input features. Therefore, no feature was removed solely on the basis of this correlation analysis.

Correlation does not establish causation and may not capture nonlinear relationships.

## 16. Chi-Square Feature Selection

Chi-Square evaluates statistical association between non-negative features and a categorical target.

It was applied to the one-hot encoded categorical features.

```python
from sklearn.feature_selection import SelectKBest, chi2

chi_selector = SelectKBest(
    score_func=chi2,
    k=20
)

X_train_chi = chi_selector.fit_transform(
    X_train_cat,
    y_train
)

X_test_chi = chi_selector.transform(
    X_test_cat
)
```

### Observed Result

| Metric | Value |
|---|---:|
| Categorical features before selection | 102 |
| Selected features | 20 |

Selected features included categories related to:

- Education
- Marital status
- Occupation
- Relationship
- Sex

Examples include `education_Bachelors`, `education_Masters`, `occupation_Exec-managerial`, `relationship_Husband`, and `sex_Male`.

### Interpretation

The selected features had the highest Chi-Square scores among the 102 encoded categorical features. A high score indicates evidence of association with the target, not necessarily high predictive performance.

## 17. Mutual Information

Mutual Information estimates the statistical dependence between an input feature and the target variable.

```python
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif
)

mi_selector = SelectKBest(
    score_func=mutual_info_classif,
    k=20
)

X_train_mi = mi_selector.fit_transform(
    X_train_var,
    y_train
)

X_test_mi = mi_selector.transform(
    X_test_var
)
```

### Observed Result

| Metric | Value |
|---|---:|
| Input features | 60 |
| Selected features | 20 |

The selected subset included numerical features such as `age`, `education-num`, `capital-gain`, and `hours-per-week`, along with selected encoded categories.

### Interpretation

Mutual Information can identify statistical dependencies that are not necessarily linear. The result depends on the estimation method and data characteristics.

## 18. ANOVA Feature Selection

ANOVA F-test evaluates whether numerical feature means differ across the target classes.

```python
from sklearn.feature_selection import SelectKBest, f_classif

anova_selector = SelectKBest(
    score_func=f_classif,
    k=4
)

X_train_anova = anova_selector.fit_transform(
    X_train[numerical_cols],
    y_train
)

X_test_anova = anova_selector.transform(
    X_test[numerical_cols]
)
```

### Observed Result

Four features were selected:

1. `age`
2. `education-num`
3. `capital-gain`
4. `hours-per-week`

The features `fnlwgt` and `capital-loss` were not among the four selected features.

### Interpretation

The selected features had the highest ANOVA F-scores among the numerical input features. This ranking does not establish that the excluded features are useless for classification.

## 19. Recursive Feature Elimination (RFE)

RFE is a wrapper-based method that repeatedly fits an estimator and removes features considered less important by that estimator.

Logistic Regression was used as the estimator.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

rfe_selector = RFE(
    estimator=rfe_model,
    n_features_to_select=20,
    step=1
)

X_train_rfe = rfe_selector.fit_transform(
    X_train_var,
    y_train
)

X_test_rfe = rfe_selector.transform(
    X_test_var
)
```

### Observed Result

| Metric | Value |
|---|---:|
| Input features | 60 |
| Selected features | 20 |

The selected features included `education-num`, `capital-gain`, and encoded categories related to workclass, education, marital status, occupation, relationship, sex, and native country.

### Interpretation

RFE selects a feature subset based on the chosen estimator. A different estimator or model configuration may select a different subset.

---

## 📈 Feature Selection Summary

| Method | Input features | Selected features | Purpose |
|---|---:|---:|---|
| Variance Threshold | 108 | 60 | Remove low-variance features |
| Correlation Analysis | 6 numerical features | No features removed | Examine linear relationships |
| Chi-Square | 102 categorical features | 20 | Evaluate categorical-feature association |
| Mutual Information | 60 | 20 | Estimate statistical dependence |
| ANOVA F-test | 6 numerical features | 4 | Compare numerical feature means across classes |
| RFE | 60 | 20 | Recursively select features using Logistic Regression |

The methods have different objectives and operate on different feature sets. A fair comparison of predictive usefulness requires evaluating the resulting models using the same validation strategy.

## 20. Model Evaluation — Next Stage

The next stage is to evaluate whether feature selection improves classification performance.

Logistic Regression can be used to compare the full feature set against selected feature subsets.

Recommended evaluation metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Counts of correct and incorrect predictions by class.

Because the income classes are imbalanced, accuracy should not be used as the only evaluation metric.

**Model evaluation results should be added after running the experiment. No performance values are reported here because the final comparison has not yet been documented.**

## ⚠️ Important Methodological Notes

1. The original dataset contains 32,561 rows, while 32,537 rows remain after duplicate removal.
2. The recorded income distribution of 24,720 and 7,841 observations refers to the dataset before duplicate removal.
3. `?` values are not automatically treated as missing values by Pandas. Their replacement or imputation strategy should be documented explicitly.
4. Outliers were identified using boxplots and the IQR method, but they were not automatically removed.
5. Encoding, scaling, and feature selection were fitted using training data and then applied to the test data.
6. Correlation and feature-selection scores describe statistical relationships; they do not establish causation.
7. For a rigorous final comparison, preprocessing and feature selection should be incorporated into a Scikit-learn Pipeline, and hyperparameter or feature-count selection should use training-only cross-validation. The test set should be reserved for final evaluation.

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/adult-income-classification.git
cd adult-income-classification
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place the Adult Census Income CSV file at the path expected by the notebook, or update the dataset path in the first notebook cell.

### 4. Run the Notebook

Open `notebooks/adult_income_analysis.ipynb` using Jupyter Notebook or Google Colab and execute the cells in order.

## 📦 Requirements

Create a `requirements.txt` file containing the following packages:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## 📚 Key Learnings

- Understanding numerical and categorical data
- Exploratory data analysis and statistical summaries
- Identifying missing values represented by special symbols
- Duplicate detection and handling
- Visualizing and investigating outliers
- Separating input features from the target variable
- Preventing data leakage during preprocessing
- One-Hot Encoding for categorical data
- StandardScaler for numerical data
- Filter-based feature selection
- Wrapper-based feature selection
- Correlation analysis and statistical feature ranking
- Preparing a dataset for binary classification

## 🔮 Future Improvements

- Build a complete preprocessing Pipeline using `ColumnTransformer`.
- Compare Logistic Regression, Decision Tree, Random Forest, and other classifiers.
- Use cross-validation to compare feature-selection strategies fairly.
- Evaluate precision, recall, F1-score, and the confusion matrix.
- Investigate class imbalance and appropriate resampling or class-weight strategies.
- Add feature-importance visualizations.
- Document the final model's performance and limitations.
- Examine potential fairness and bias concerns when using demographic attributes.

## 👨‍💻 Author

**Harshit**

Machine Learning Assignment — Data Preprocessing and Feature Selection

## 📄 License

This project was created for educational purposes. Add an appropriate license if you intend to distribute or reuse the repository publicly.
