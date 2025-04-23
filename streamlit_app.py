import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    accuracy_score, precision_score,
    recall_score, f1_score
)

# ---------------------------
# Page configuration & style
# ---------------------------
st.set_page_config(
    page_title="Bankruptcy Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.main-header { font-size: 2.5rem; font-weight: bold; color: #395c40; }
.sub-header  { font-size: 1.5rem; font-weight: bold; color: #395c40; margin-top:1rem; }
.section-header { font-size: 1.2rem; font-weight: bold; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Bankruptcy Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown("""
This dashboard presents a comprehensive analysis of bankruptcy prediction models using financial data 
from American companies. Models are trained on 1999–2011 and tested on 2015–2018 data.
""")

# ---------------------------
# Data loading & preprocessing
# ---------------------------
rename_map = {
    "X1": "Current Assets", "X2": "Cost of Goods Sold", "X3": "D&A",
    "X4": "EBITDA", "X5": "Inventory", "X6": "Net Income",
    "X7": "Total Receivables", "X8": "Market Value", "X9": "Net Sales",
    "X10":"Total Assets","X11":"Total Long-term Debt","X12":"EBIT",
    "X13":"Gross Profit","X14":"Total Current Liabilities",
    "X15":"Retained Earnings","X16":"Total Revenue",
    "X17":"Total Liabilities","X18":"Total Operating Expenses"
}

@st.cache_data
def load_and_prepare():
    # try multiple paths
    paths = [
        "data/american_bankruptcy.csv",
        "american_bankruptcy.csv",
        "../data/american_bankruptcy.csv",
        "./american_bankruptcy.csv"
    ]
    df = pd.DataFrame()
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    if df.empty:
        st.error("❌ Could not find 'american_bankruptcy.csv'")
        return df
    # map status_label → Bankruptcy
    if "status_label" in df.columns:
        df["Bankruptcy"] = df["status_label"].map({"failed":1,"alive":0})
    elif "Bankruptcy" not in df.columns:
        st.error("❌ No 'status_label' or 'Bankruptcy' column")
        return df
    # rename X1–X18
    df = df.rename(columns=rename_map)
    return df

data = load_and_prepare()
st.session_state["data_loaded"] = not data.empty

# train/test split
if st.session_state["data_loaded"]:
    train = data[(data.year>=1999)&(data.year<=2011)]
    test  = data[(data.year>=2015)&(data.year<=2018)]
    features = list(rename_map.values())
    X_train, y_train = train[features], train["Bankruptcy"]
    X_test,  y_test  = test[features],  test["Bankruptcy"]

# ---------------------------
# Model training & metrics
# ---------------------------
models = {}

if st.session_state["data_loaded"]:
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train,y_train)
    models["Decision Tree"] = dt

    # Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train,y_train)
    models["Gradient Boosting"] = gb

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train,y_train)
    models["Random Forest"] = rf

    # Logistic Regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(solver="liblinear", random_state=42))
    ])
    lr_pipe.fit(X_train,y_train)
    models["Logistic Regression"] = lr_pipe

    # SVM
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, random_state=42))
    ])
    svm_pipe.fit(X_train,y_train)
    models["SVM"] = svm_pipe

    # KNN
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])
    knn_pipe.fit(X_train,y_train)
    models["KNN"] = knn_pipe

# compute metrics for each model
metrics = {}
roc_data = {}
if st.session_state["data_loaded"]:
    for name, mdl in models.items():
        y_pred  = mdl.predict(X_test)
        y_proba = mdl.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr,tpr)
        cm = confusion_matrix(y_test,y_pred)
        metrics[name] = {
            "accuracy": accuracy_score(y_test,y_pred),
            "precision": precision_score(y_test,y_pred),
            "recall": recall_score(y_test,y_pred),
            "f1": f1_score(y_test,y_pred),
            "auc": roc_auc,
            "confusion_matrix": cm.tolist()
        }
        roc_data[name] = (fpr,tpr,roc_auc)

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("Navigation")
pages = [
    "Overview",
    "Model Comparison",
    "ROC Curves",
    "Feature Importance",
    "Confusion Matrices",
    "Altman Z-Score"
]
page = st.sidebar.radio("Go to", pages)

# ---------------------------
# Overview
# ---------------------------
if page=="Overview":
    st.markdown('<p class="sub-header">Overview</p>', unsafe_allow_html=True)
    st.markdown("""
    - **Train**: 1999–2011  
    - **Test**: 2015–2018  
    - **Features**: 18 financial metrics  
    - **Target**: Bankruptcy (0=Alive, 1=Bankrupt)
    """)
    if st.session_state["data_loaded"]:
        dfm = pd.DataFrame(metrics).T[["accuracy","precision","recall","f1","auc"]]
        dfm.columns = ["Accuracy","Precision","Recall","F1 Score","AUC"]
        c1,c2,c3 = st.columns(3)
        c1.metric("Best AUC", f"{dfm['AUC'].max():.3f}", dfm['AUC'].idxmax())
        c2.metric("Best F1 Score", f"{dfm['F1 Score'].max():.3f}", dfm['F1 Score'].idxmax())
        c3.metric("Best Recall", f"{dfm['Recall'].max():.3f}", dfm['Recall'].idxmax())
        st.markdown("### Model AUC Comparison")
        aucs = dfm["AUC"].sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(8,4))
        bars=ax.bar(aucs.index,aucs.values,color='#395c40')
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f"{b.get_height():.3f}",ha='center',va='bottom')
        ax.set_ylabel("AUC"); ax.set_xticklabels(aucs.index, rotation=45, ha='right')
        st.pyplot(fig)

        # Dataset stats & bankruptcy distribution
        st.markdown("### Dataset Statistics")
        st.write(f"Records: {len(data)}, Features: {len(data.columns)}")
        if "Bankruptcy" in data.columns:
            cnt = data["Bankruptcy"].value_counts().rename({0:"Healthy",1:"Bankrupt"})
            st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)
            fig,ax=plt.subplots(figsize=(6,3))
            bars=ax.bar(cnt.index,cnt.values,color=['#395c40','#a63603'])
            mx=cnt.values.max()
            ax.set_ylim(0,mx*1.2); ax.set_ylabel("Count")
            for i,v in enumerate(cnt.values):
                ax.text(i, v+mx*0.05, f"{v}", ha='center', va='bottom', fontsize=14)
            st.pyplot(fig)
            st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)

# ---------------------------
# Model Comparison
# ---------------------------
elif page=="Model Comparison":
    st.markdown('<p class="sub-header">Model Comparison</p>', unsafe_allow_html=True)
    if st.session_state["data_loaded"]:
        dfm = pd.DataFrame(metrics).T[["accuracy","precision","recall","f1","auc"]]
        dfm.columns = ["Accuracy","Precision","Recall","F1 Score","AUC"]
        st.dataframe(dfm.style.highlight_max(axis=0))
        opts=["Accuracy","Precision","Recall","F1 Score","AUC"]
        sel = st.multiselect("Select metrics", opts, default=["Recall","F1 Score","AUC"])
        if sel:
            fig,axs=plt.subplots(1,len(sel),figsize=(4*len(sel),4))
            if len(sel)==1: axs=[axs]
            for i,m in enumerate(sel):
                d = dfm.sort_values(m,ascending=False)
                bars = axs[i].bar(d.index,d[m],color='#395c40')
                for b in bars:
                    axs[i].text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f"{b.get_height():.3f}",ha='center',va='bottom')
                axs[i].set_title(m); axs[i].tick_params(axis='x', rotation=45)
            st.pyplot(fig)

# ---------------------------
# ROC Curves
# ---------------------------
elif page=="ROC Curves":
    st.markdown('<p class="sub-header">ROC Curves</p>', unsafe_allow_html=True)
    if st.session_state["data_loaded"]:
        mods = list(models.keys())
        sel = st.multiselect("Select models", mods, default=mods[:3])
        if sel:
            fig,ax = plt.subplots(figsize=(6,6))
            ax.plot([0,1],[0,1],'--',color='gray',alpha=0.7)
            for m in sel:
                fpr,tpr,aucv = roc_data[m]
                ax.plot(fpr,tpr, label=f"{m} (AUC={aucv:.3f})")
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right"); st.pyplot(fig)

# ---------------------------
# Feature Importance
# ---------------------------
elif page=="Feature Importance":
    st.markdown('<p class="sub-header">Feature Importance</p>', unsafe_allow_html=True)
    if st.session_state["data_loaded"]:
        mods = ["Decision Tree","Gradient Boosting","Random Forest","Logistic Regression","SVM","KNN"]
        sel = st.selectbox("Model", mods)
        # extract importances / coefficients / permutation
        if sel in ["Decision Tree","Gradient Boosting","Random Forest"]:
            imp = pd.Series(models[sel].feature_importances_, index=features).sort_values(ascending=False)
        elif sel=="Logistic Regression":
            coef = models[sel].named_steps["lr"].coef_[0]
            imp = pd.Series(np.abs(coef), index=features).sort_values(ascending=False)
        else:
            # SVM & KNN: use permutation_importance
            from sklearn.inspection import permutation_importance
            res = permutation_importance(models[sel], X_test, y_test, n_repeats=5, random_state=0)
            imp = pd.Series(res.importances_mean, index=features).sort_values(ascending=False)
        topn = st.slider("Top features to display", 5, len(features), 10)
        top = imp.head(topn)
        fig,ax = plt.subplots(figsize=(6,topn*0.3))
        bars = ax.barh(top.index[::-1], top.values[::-1], color='#395c40')
        mx = imp.max()
        for b in bars:
            ax.text(b.get_width()+mx*0.02, b.get_y()+b.get_height()/2, f"{b.get_width():.3f}", va='center')
        ax.set_xlabel("Importance"); st.pyplot(fig)

# ---------------------------
# Confusion Matrices
# ---------------------------
elif page=="Confusion Matrices":
    st.markdown('<p class="sub-header">Confusion Matrices</p>', unsafe_allow_html=True)
    if st.session_state["data_loaded"]:
        sel = st.selectbox("Select model", list(models.keys()))
        cm = np.array(metrics[sel]["confusion_matrix"])
        tn,fp = cm[0]; fn,tp = cm[1]
        # show table
        df_cm = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])
        st.dataframe(df_cm)
        # metrics
        st.markdown(f"- Accuracy: {metrics[sel]['accuracy']:.4f}")
        st.markdown(f"- Precision: {metrics[sel]['precision']:.4f}")
        st.markdown(f"- Recall: {metrics[sel]['recall']:.4f}")
        st.markdown(f"- F1 Score: {metrics[sel]['f1']:.4f}")

# ---------------------------
# Altman Z-Score Tab
# ---------------------------
elif page=="Altman Z-Score":
    st.markdown('<p class="sub-header">Altman Z-Score Analysis</p>', unsafe_allow_html=True)
    st.markdown("""
    The Altman Z-Score predicts bankruptcy risk via:
    Z = 1.2·T1 + 1.4·T2 + 3.3·T3 + 0.6·T4 + 0.99·T5  
    where:
    - T1 = (Current Assets − Total Current Liabilities)/Total Assets  
    - T2 = Retained Earnings/Total Assets  
    - T3 = EBIT/Total Assets  
    - T4 = Market Value/Total Liabilities  
    - T5 = Net Sales/Total Assets  

    Zones:
    - **Distress**: Z < 1.80  
    - **Grey**:     1.80 ≤ Z ≤ 2.99  
    - **Safe**:    Z > 2.99
    """)
    if st.session_state["data_loaded"]:
        # calculate Z-score
        df = data.copy()
        z = pd.DataFrame(index=df.index)
        z["T1"] = (df["Current Assets"] - df["Total Current Liabilities"]) / df["Total Assets"]
        z["T2"] = df["Retained Earnings"] / df["Total Assets"]
        z["T3"] = df["EBIT"] / df["Total Assets"]
        z["T4"] = df["Market Value"] / df["Total Liabilities"]
        z["T5"] = df["Net Sales"] / df["Total Assets"]
        z["Z-Score"] = (
            1.2*z["T1"] + 1.4*z["T2"] +
            3.3*z["T3"] + 0.6*z["T4"] + 0.99*z["T5"]
        )
        z["Z-Score"].replace([np.inf,-np.inf],np.nan,inplace=True)
        z["Z-Score"].fillna(z["Z-Score"].median(), inplace=True)
        bins=[-np.inf,1.8,2.99,np.inf]
        labels=["Distress","Grey","Safe"]
        z["Zone"] = pd.cut(z["Z-Score"], bins=bins, labels=labels)
        z["Pred"] = (z["Zone"]=="Distress").astype(int)
        z["Actual"] = df["Bankruptcy"].astype(int)

        # confusion matrix & metrics
        cm = confusion_matrix(z["Actual"], z["Pred"])
        tn,fp = cm[0]; fn,tp = cm[1]
        acc = (tn+tp)/cm.sum()
        prec = tp/(tp+fp) if tp+fp>0 else 0
        rec = tp/(tp+fn) if tp+fn>0 else 0
        f1v = 2*prec*rec/(prec+rec) if prec+rec>0 else 0

        st.markdown("#### Z-Score vs. Machine Learning")
        df_compare = pd.DataFrame(metrics).T[["accuracy","precision","recall","f1"]]
        df_compare.loc["Altman Z-Score"] = [acc,prec,rec,f1v]
        df_compare.columns = ["Accuracy","Precision","Recall","F1 Score"]
        st.dataframe(df_compare.style.highlight_max(axis=0))

        st.markdown("#### Altman Z-Score Confusion Matrix")
        st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

        st.markdown(f"- **Accuracy**: {acc:.4f}")
        st.markdown(f"- **Precision**: {prec:.4f}")
        st.markdown(f"- **Recall**: {rec:.4f}")
        st.markdown(f"- **F1 Score**: {f1v:.4f}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.8em;">
Bankruptcy Prediction Dashboard | © 2025
</div>
""", unsafe_allow_html=True)
