
# tabs.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

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

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.3, random_state: int = 42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 2. Modeling utilities
def train_models(X_train, y_train) -> dict:
    # Ensure numeric only and fill missing
    X_train = X_train.select_dtypes(include=['number']).fillna(0)
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }
    pipelines = {}
    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
    return pipelines

def get_metrics(pipelines: dict, X_test, y_test) -> dict:
    # Ensure numeric only and fill missing
    X_test = X_test.select_dtypes(include=['number']).fillna(0)
    results = {}
    for name, pipe in pipelines.items():
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_score,
            "cm": cm,
            "cr": cr
        }
    return results

def get_feature_importances(models: dict, feature_names: list) -> dict:
    imps = {}
    for name, pipe in models.items():
        if hasattr(pipe['clf'], 'feature_importances_'):
            importances = pipe['clf'].feature_importances_
            imps[name] = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    return imps

def get_permutation_importances(models: dict, X_test, y_test, feature_names: list) -> dict:
    X_test = X_test.select_dtypes(include=['number']).fillna(0)
    imps = {}
    for name, pipe in models.items():
        result = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=42)
        imps[name] = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
    return imps

# 3. Tab views
def show_overview_tab(df: pd.DataFrame):
    st.subheader("Dataset Overview")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.subheader("Features")
    st.write(", ".join([col for col in df.columns if col != "Bankruptcy"]))
    st.dataframe(df.head())
    
    # Show bankruptcy distribution
    st.subheader("Bankruptcy Distribution")
    bankruptcy_counts = df["Bankruptcy"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(bankruptcy_counts, labels=["Alive", "Bankrupt"], autopct='%1.1f%%')
    st.pyplot(fig)

def show_training_tab(results: dict):
    st.subheader("Model Performance Metrics")
    df_metrics = pd.DataFrame({
        name: {
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1 Score": r["f1"],
            "AUC": r["auc"]
        }
        for name, r in results.items()
    }).T
    df_metrics = df_metrics.applymap(lambda x: f"{x:.3f}")
    st.dataframe(df_metrics)

def show_comparison_tab(results: dict):
    st.subheader("Model AUC Comparison")
    names = list(results.keys())
    aucs = [r["auc"] for r in results.values()]
    fig, ax = plt.subplots()
    ax.bar(names, aucs, edgecolor='black')
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
        ax.set_xticklabels(["Alive","Bankrupt"])
        ax.set_yticklabels(["Alive","Bankrupt"])
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j]>thresh else "black")
        st.pyplot(fig)

def show_classification_tab(results: dict):
    st.subheader("Classification Reports")
    for name, r in results.items():
        df_cr = pd.DataFrame(r["cr"]).T
        st.write(f"**{name}**")
        st.dataframe(df_cr)

def show_feature_importance_tab(imps: dict):
    st.subheader("Feature Importances (Tree Models)")
    for name, series in imps.items():
        st.write(f"**{name}**")
        fig, ax = plt.subplots(figsize=(6,4))
        series.head(10).plot.bar(ax=ax)
        ax.set_ylabel("Importance")
        ax.set_title(f"{name} Top 10 Features")
        st.pyplot(fig)

def show_permutation_tab(imps: dict):
    st.subheader("Permutation Importances")
    for name, series in imps.items():
        st.write(f"**{name}**")
        fig, ax = plt.subplots(figsize=(6,4))
        series.head(10).plot.bar(ax=ax)
        ax.set_ylabel("Importance")
        ax.set_title(f"{name} Top 10 Permutation Importances")
        st.pyplot(fig)
