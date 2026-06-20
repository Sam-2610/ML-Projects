# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved based on applicant financial and demographic data. Compares three classification models and evaluates the impact of feature engineering on performance.

## Overview

This project walks through a complete ML workflow:

1. Data cleaning and missing value imputation
2. Exploratory data analysis (EDA) with visualizations
3. Categorical encoding (label + one-hot)
4. Correlation analysis
5. Model training and evaluation (Logistic Regression, KNN, Naive Bayes)
6. Feature engineering and re-evaluation to measure improvement

## Dataset

The script expects a CSV file named `loan_approved_data.csv` in the project root, with the following columns:

| Column              | Description                                 |
| ------------------- | ------------------------------------------- |
| `Applicant_ID`      | Unique identifier (dropped before modeling) |
| `Gender`            | Applicant gender                            |
| `Education_Level`   | Applicant education level                   |
| `Employment_Status` | Employment status                           |
| `Marital_Status`    | Marital status                              |
| `Loan_Purpose`      | Purpose of the loan                         |
| `Property_Area`     | Urban / Semiurban / Rural                   |
| `Employer`          | Applicant's employer                        |
| `Applicant_Income`  | Applicant's income                          |
| `Credit_Score`      | Applicant's credit score                    |
| `DTI_Ratio`         | Debt-to-income ratio                        |
| `Savings`           | Applicant's savings                         |
| `Loan_Approved`     | Target variable (Yes/No)                    |

> The dataset itself is not included in this repository. Add your own CSV with matching column names, or update `DATA_PATH` in the script.

## Project Structure

```
.
├── loan_approval_prediction.py   # Main script
├── requirements.txt              # Python dependencies
├── plots/                        # Generated EDA charts (created on run)
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/loan-approval-prediction.git
cd loan-approval-prediction
pip install -r requirements.txt
```

Place `loan_approved_data.csv` in the project root, then run:

```bash
python loan_approval_prediction.py
```

## Workflow

### 1. Missing Value Handling

- Numerical columns imputed with the **mean**
- Categorical columns imputed with the **most frequent value**

### 2. Exploratory Data Analysis

Generates and saves the following plots to `plots/`:

- Loan approval class balance (pie chart)
- Gender distribution (bar chart)
- Income distribution (histogram)
- Outlier detection across income, credit score, DTI ratio, and savings (box plots)
- Credit score and income vs. loan approval (dodge histograms)
- Feature correlation heatmap

### 3. Encoding

- `LabelEncoder` for `Education_Level` and the target `Loan_Approved`
- `OneHotEncoder` for nominal columns: `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer`

### 4. Modeling

Three classifiers are trained and compared on an 80/20 train-test split, with features standardized via `StandardScaler`:

- **Logistic Regression**
- **K-Nearest Neighbors** (k=5)
- **Gaussian Naive Bayes**

Each model is evaluated with accuracy, precision, recall, F1 score, and a confusion matrix.

### 5. Feature Engineering

Adds squared terms for `Credit_Score` and `DTI_Ratio` to capture potential non-linear relationships, then re-trains and re-evaluates all three models to compare against the baseline.

## Results

The script prints a side-by-side comparison table of all three models, before and after feature engineering, so you can see whether the added features improved performance. Exact metrics depend on the dataset used.

## Tech Stack

- **pandas / numpy** — data manipulation
- **seaborn / matplotlib** — visualization
- **scikit-learn** — preprocessing, modeling, evaluation

## Possible Next Steps

- Hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`)
- Cross-validation instead of a single train/test split
- Additional models (Random Forest, XGBoost)
- Handling class imbalance if present (e.g. `class_weight`, SMOTE)

## Author

**Satyam Sagar**
satyamsagar@gmail.com

## License

This project is open source and available under the [MIT License](LICENSE).
