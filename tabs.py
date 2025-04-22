import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

@st.cache_data
def load_data(path: str = 'data/bankruptcy_data.csv') -> pd.DataFrame:
    """Read dataset from local CSV file."""
    return pd.read_csv(path)

def show_data(df: pd.DataFrame):
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape}")
    st.subheader("Feature Distributions")
    for col in df.select_dtypes(include=['number']).columns:
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

def train_models(df: pd.DataFrame, target: str) -> dict:
    X = df.drop(columns=[target])
    y = df[target]
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier()
    }
    results = {}
    for name, model in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        pipe.fit(X, y)
        probs = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        fpr, tpr, _ = roc_curve(y, probs)
        results[name] = {'model': pipe, 'auc': auc, 'fpr': fpr, 'tpr': tpr}
    return results

def show_training(results: dict):
    st.subheader("Training Results (AUC)")
    for name, res in results.items():
        st.write(f"**{name}**: {res['auc']:.3f}")

def show_comparison(results: dict):
    st.subheader("Model Comparison: AUC Scores")
    names = list(results.keys())
    aucs = [results[name]['auc'] for name in names]
    fig, ax = plt.subplots()
    ax.bar(names, aucs)
    ax.set_ylabel('AUC')
    ax.set_ylim(0,1)
    ax.set_title('Model AUC Comparison')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

def show_evaluation(results: dict):
    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, res in results.items():
        ax.plot(res['fpr'], res['tpr'], label=f"{name} ({res['auc']:.3f})")
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Comparison')
    ax.legend()
    st.pyplot(fig)

def show_prediction(results: dict, features: list):
    st.subheader("Make a New Prediction")
    inputs = {feat: st.number_input(feat, value=0.0) for feat in features}
    model_name = st.selectbox("Select Model", list(results.keys()))
    if st.button("Predict Probability"):
        df_new = pd.DataFrame([inputs])
        prob = results[model_name]['model'].predict_proba(df_new)[0,1]
        st.write(f"Predicted bankruptcy probability: {prob:.2%}")
