import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Data utilities
@st.cache_data
def load_data(path: str = "data/american_bankruptcy.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Bankruptcy"] = df["status_label"].map({"failed": 1, "alive": 0})
    rename_map = {
        "X1": "Current Assets", "X2": "Cost of Goods Sold",
        "X3": "D&A", "X4": "EBITDA", "X5": "Inventory",
        "X6": "Net Income", "X7": "Total Receivables",
        "X8": "Market Value", "X9": "Net Sales",
        "X10": "Total Assets", "X11": "Total Long-term Debt",
        "X12": "EBIT", "X13": "Gross Profit",
        "X14": "Total Current Liabilities", "X15": "Retained Earnings",
        "X16": "Total Revenue", "X17": "Total Liabilities",
        "X18": "Total Operating Expenses"
    }
    df = df.rename(columns=rename_map)
    return df.drop(columns=["status_label"])

def split_data(df: pd.DataFrame, target: str, test_size: float = 0.3, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 2. Modeling utilities
def train_models(X_train, y_train) -> dict:
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }
    fitted = {}
    for name, m in models.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", m)])
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
    return fitted

def get_metrics(models: dict, X_test, y_test) -> dict:
    results = {}
    for name, m in models.items():
        y_pred = m.predict(X_test)
        y_proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            "fpr": fpr, "tpr": tpr, "auc": auc_score,
            "cm": cm, "cr": cr
        }
    return results

# 3. Tab views
def show_data_tab(df: pd.DataFrame):
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape}")
    st.subheader("Feature Distributions")
    for col in df.select_dtypes(include=["number"]).columns:
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(col)
        st.pyplot(fig)

def show_training_tab(results: dict):
    st.subheader("Training AUC Scores")
    for name, r in results.items():
        st.write(f"{name}: {r['auc']:.3f}")

def show_comparison_tab(results: dict):
    st.subheader("Model AUC Comparison")
    names = list(results.keys())
    aucs = [r['auc'] for r in results.values()]
    fig, ax = plt.subplots()
    ax.bar(names, aucs)
    ax.set_ylim(0,1)
    ax.set_ylabel("AUC")
    ax.set_xticklabels(names, rotation=45, ha="right")
    st.pyplot(fig)

def show_roc_tab(results: dict):
    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, r in results.items():
        ax.plot(r["fpr"], r["tpr"], label=f"{name} ({r['auc']:.3f})")
    ax.plot([0,1], [0,1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

def show_confusion_tab(results: dict):
    st.subheader("Confusion Matrices")
    for name, r in results.items():
        cm = r["cm"]
        fig, ax = plt.subplots(figsize=(4,3))
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Alive","Bankrupt"]); ax.set_yticklabels(["Alive","Bankrupt"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i,j], ha="center", va="center")
        st.pyplot(fig)

def show_classification_tab(results: dict):
    st.subheader("Classification Reports")
    for name, r in results.items():
        df_cr = pd.DataFrame(r["cr"]).T
        st.write(f"**{name}**")
        st.dataframe(df_cr)

def show_prediction_tab(models: dict, features: list):
    st.subheader("Make a New Prediction")
    inputs = {feat: st.number_input(feat, value=0.0) for feat in features}
    model_name = st.selectbox("Select Model", list(models.keys()))
    if st.button("Predict"):
        proba = models[model_name].predict_proba(pd.DataFrame([inputs]))[0,1]
        st.write(f"Predicted Bankruptcy Probability: {proba:.1%}")
