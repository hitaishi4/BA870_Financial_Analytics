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
def load_data(path):
    return pd.read_csv(path)

def show_data(df):
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape}")
    st.subheader("Feature Distributions")
    for col in df.select_dtypes(['float64', 'int64']).columns:
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

@st.cache_resource
def train_models(df, target):
    X, y = df.drop(columns=[target]), df[target]
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

def show_training(results):
    st.subheader("Training Results")
    for name, res in results.items():
        st.write(f"**{name}**: AUC = {res['auc']:.3f}")

def show_evaluation(results):
    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, res in results.items():
        ax.plot(res['fpr'], res['tpr'], label=f"{name} ({res['auc']:.3f})")
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    st.pyplot(fig)

def show_prediction(results, features):
    st.subheader("Make a Prediction")
    inp = {f: st.number_input(f, value=0.0) for f in features}
    model_name = st.selectbox("Model", list(results.keys()))
    if st.button("Predict Bankruptcy Probability"):
        prob = results[model_name]['model'].predict_proba(pd.DataFrame([inp]))[0,1]
        st.write(f"Predicted probability of bankruptcy: {prob:.2%}")

def main():
    st.title("Bankruptcy Prediction App")
    uploaded = st.file_uploader("Upload CSV data", type=['csv'])
    if not uploaded:
        st.info("Please upload a CSV file to proceed.")
        return
    df = load_data(uploaded)
    target = st.text_input("Target column name", value='bankrupt')
    tabs = st.tabs(["Data", "Training", "Evaluation", "Prediction"])
    with tabs[0]:
        show_data(df)
    with tabs[1]:
        if st.button("Train Models"):
            st.session_state['results'] = train_models(df, target)
            show_training(st.session_state['results'])
    with tabs[2]:
        if 'results' in st.session_state:
            show_evaluation(st.session_state['results'])
        else:
            st.info("Please train models first.")
    with tabs[3]:
        if 'results' in st.session_state:
            features = df.drop(columns=[target]).columns.tolist()
            show_prediction(st.session_state['results'], features)
        else:
            st.info("Please train models first.")

if __name__ == "__main__":
    main()
