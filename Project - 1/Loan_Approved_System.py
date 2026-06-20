"""
Loan Approval Prediction
=========================
Predicts whether a loan application will be approved using applicant
financial and demographic data. Compares Logistic Regression, KNN, and
Naive Bayes classifiers, then evaluates the impact of feature engineering
on model performance.

Dataset columns expected:
    Applicant_ID, Gender, Education_Level, Employment_Status,
    Marital_Status, Loan_Purpose, Property_Area, Employer,
    Applicant_Income, Credit_Score, DTI_Ratio, Savings, Loan_Approved
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

RANDOM_STATE = 42
DATA_PATH = "loan_approved_data.csv"  # place the CSV in the same folder as this script


# ---------------------------------------------------------------------------
# Helper: train + evaluate any classifier with consistent, readable output
# ---------------------------------------------------------------------------
def evaluate_model(name, model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def run_all_models(x_train, y_train, x_test, y_test, label=""):
    """Trains Logistic Regression, KNN, and Naive Bayes and returns a results table."""
    if label:
        print(f"\n{'=' * 50}\n{label}\n{'=' * 50}")

    results = []
    results.append(evaluate_model(
        "Logistic Regression", LogisticRegression(),
        x_train, y_train, x_test, y_test
    ))
    results.append(evaluate_model(
        "K-Nearest Neighbors (k=5)", KNeighborsClassifier(n_neighbors=5),
        x_train, y_train, x_test, y_test
    ))
    results.append(evaluate_model(
        "Naive Bayes", GaussianNB(),
        x_train, y_train, x_test, y_test
    ))
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print(df.head())
print(df.info())
print("Missing values per column:\n", df.isnull().sum())
print(df.describe())


# ---------------------------------------------------------------------------
# 2. Handle Missing Values
# ---------------------------------------------------------------------------
categorical_columns = df.select_dtypes(include=["object"]).columns
numerical_columns = df.select_dtypes(include=["float64"]).columns

num_imputer = SimpleImputer(strategy="mean")
df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])

cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])


# ---------------------------------------------------------------------------
# 3. Exploratory Data Analysis
# ---------------------------------------------------------------------------

# Class balance
classes_count = df["Loan_Approved"].value_counts()
plt.figure()
plt.pie(classes_count, labels=["No", "Yes"], autopct="%1.1f%%")
plt.title("Is Loan Approved or Not")
plt.savefig("plots/class_balance.png", bbox_inches="tight")
plt.show()

# Gender distribution
plt.figure()
gender_cnt = df["Gender"].value_counts()
ax = sns.barplot(x=gender_cnt.index, y=gender_cnt.values)
ax.bar_label(ax.containers[0])
plt.title("Applicant Gender Distribution")
plt.savefig("plots/gender_distribution.png", bbox_inches="tight")
plt.show()

# Income distribution
plt.figure()
sns.histplot(data=df, x="Applicant_Income", bins=20)
plt.title("Applicant Income Distribution")
plt.savefig("plots/income_distribution.png", bbox_inches="tight")
plt.show()

# Outlier check: income vs loan approval
plt.figure()
sns.boxplot(data=df, x="Loan_Approved", y="Applicant_Income")
plt.title("Applicant Income by Loan Approval")
plt.savefig("plots/income_vs_approval_boxplot.png", bbox_inches="tight")
plt.show()

# Outlier check across all key numeric features
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
sns.boxplot(ax=axes[0, 0], data=df, x="Loan_Approved", y="Applicant_Income")
sns.boxplot(ax=axes[0, 1], data=df, x="Loan_Approved", y="Credit_Score")
sns.boxplot(ax=axes[1, 0], data=df, x="Loan_Approved", y="DTI_Ratio")
sns.boxplot(ax=axes[1, 1], data=df, x="Loan_Approved", y="Savings")
plt.tight_layout()
plt.savefig("plots/feature_boxplots.png", bbox_inches="tight")
plt.show()

# Credit score vs loan approval
plt.figure()
sns.histplot(
    data=df,
    x="Credit_Score",
    hue="Loan_Approved",
    bins=20,
    multiple="dodge",
)
plt.title("Credit Score by Loan Approval")
plt.savefig("plots/credit_score_vs_approval.png", bbox_inches="tight")
plt.show()

# Income vs loan approval
plt.figure()
sns.histplot(
    data=df,
    x="Applicant_Income",
    hue="Loan_Approved",
    bins=20,
    multiple="dodge",
)
plt.title("Applicant Income by Loan Approval")
plt.savefig("plots/income_vs_approval.png", bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 4. Drop Identifier Column
# ---------------------------------------------------------------------------
df = df.drop("Applicant_ID", axis=1)


# ---------------------------------------------------------------------------
# 5. Encoding
# ---------------------------------------------------------------------------

# Label encode ordinal / binary columns
le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

# One-hot encode nominal columns
cols_to_encode = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                   "Property_Area", "Gender", "Employer"]
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[cols_to_encode])
encoded_df = pd.DataFrame(
    encoded, columns=ohe.get_feature_names_out(cols_to_encode), index=df.index
)
df = pd.concat([df.drop(columns=cols_to_encode), encoded_df], axis=1)


# ---------------------------------------------------------------------------
# 6. Correlation Analysis
# ---------------------------------------------------------------------------
num_cols = df.select_dtypes(include="number")
corr_matrix = num_cols.corr()

print("\nCorrelation with Loan_Approved:")
print(corr_matrix["Loan_Approved"].sort_values(ascending=False))

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("plots/correlation_heatmap.png", bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 7. Baseline Models (original features)
# ---------------------------------------------------------------------------
x = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

baseline_results = run_all_models(
    x_train_scaled, y_train, x_test_scaled, y_test,
    label="Baseline Models (Original Features)"
)


# ---------------------------------------------------------------------------
# 8. Feature Engineering + Re-evaluation
# ---------------------------------------------------------------------------
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2

x = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
y = df["Loan_Approved"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

engineered_results = run_all_models(
    x_train_scaled, y_train, x_test_scaled, y_test,
    label="Models After Feature Engineering (squared Credit_Score & DTI_Ratio)"
)


# ---------------------------------------------------------------------------
# 9. Final Comparison
# ---------------------------------------------------------------------------
baseline_results["stage"] = "baseline"
engineered_results["stage"] = "engineered"
comparison = pd.concat([baseline_results, engineered_results], ignore_index=True)

print("\n" + "=" * 50)
print("Final Comparison: Baseline vs Feature-Engineered")
print("=" * 50)
print(comparison.to_string(index=False))