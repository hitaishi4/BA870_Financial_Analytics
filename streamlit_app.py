import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Bankruptcy Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem !important;
    font-weight: bold;
    color: #395c40;
    text-align: center;
    padding-bottom: 25px;
}
.page-header {
    font-size: 2.8rem !important;
    font-weight: bold;
    color: #395c40;
    text-align: center;
    padding-bottom: 20px;
    margin-top: 20px;
    margin-bottom: 30px;
}
.sub-header {
    font-size: 2rem !important;
    font-weight: bold;
    color: #395c40;
}
.section-header {
    font-size: 1.5rem !important;
    font-weight: bold;
}
/* Table styling */
table {
    color: black !important;
    font-weight: 700 !important;
}
th {
    color: black !important;
    font-weight: 900 !important;
}
td {
    font-weight: 700 !important;
}
.dataframe {
    font-size: 1.1rem !important;
}
.dataframe th {
    background-color: #f0f2f6;
}
.stDataFrame {
    border: 1px solid #e6e9ef;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
.sidebar-nav {
    margin-top: 1rem;
    padding-top: 2rem;
}
/* Animation */
.main-content {
    animation: fadeIn 0.5s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# Column rename mapping
rename_map = {
    "X1":  "Current Assets",
    "X2":  "Cost of Goods Sold",
    "X3":  "D&A",
    "X4":  "EBITDA",
    "X5":  "Inventory",
    "X6":  "Net Income",
    "X7":  "Total Receivables",
    "X8":  "Market Value",
    "X9":  "Net Sales",
    "X10": "Total Assets",
    "X11": "Total Long-term Debt",
    "X12": "EBIT",
    "X13": "Gross Profit",
    "X14": "Total Current Liabilities",
    "X15": "Retained Earnings",
    "X16": "Total Revenue",
    "X17": "Total Liabilities",
    "X18": "Total Operating Expenses"
}

# Data loading with caching
@st.cache_data(show_spinner=False)
def load_data():
    """Load data from multiple possible paths, apply initial mappings."""
    possible_paths = [
        Path("data/american_bankruptcy.csv"),
        Path("american_bankruptcy.csv"),
        Path("../data/american_bankruptcy.csv"),
        Path("./american_bankruptcy.csv"),
    ]
    for p in possible_paths:
        try:
            if p.exists():
                df = pd.read_csv(p)
                # Map status_label to Bankrupt
                if "status_label" in df.columns:
                    vals = df["status_label"].unique()
                    if "failed" in vals:
                        df["Bankrupt"] = df["status_label"].map({"failed": 1, "alive": 0})
                    elif "Bankrupt" in vals:
                        df["Bankrupt"] = df["status_label"].map({"Bankrupt": 1, "Alive": 0})
                    else:
                        df["Bankrupt"] = df["status_label"].apply(
                            lambda x: 1 if str(x).lower() in ["failed", "bankrupt", "distress", "default"] else 0
                        )
                # Fallback if a Bankruptcy column exists
                if "Bankruptcy" in df.columns and "Bankrupt" not in df.columns:
                    df["Bankrupt"] = df["Bankruptcy"]
                # Rename financial columns
                if "X1" in df.columns:
                    df = df.rename(columns=rename_map)
                return df
        except Exception:
            continue
    return pd.DataFrame()

# Load data
data = load_data()
st.session_state["data_loaded"] = not data.empty

# Define feature names
feature_names = list(rename_map.values())

# Performance metrics for each model
metrics = {
    'Decision Tree': {
        'accuracy': 0.8925, 'precision': 0.0589, 'recall': 0.2404,
        'f1': 0.0947, 'auc': 0.574, 'confusion_matrix': [[10893, 1102], [218, 69]]
    },
    'Gradient Boosting': {
        'accuracy': 0.9761, 'precision': 0.3846, 'recall': 0.0348,
        'f1': 0.0639, 'auc': 0.827, 'confusion_matrix': [[11979, 16], [277, 10]]
    },
    'Random Forest': {
        'accuracy': 0.9759, 'precision': 0.3200, 'recall': 0.0279,
        'f1': 0.0513, 'auc': 0.835, 'confusion_matrix': [[11978, 17], [279, 8]]
    },
    'Logistic Regression': {
        'accuracy': 0.9752, 'precision': 0.3125, 'recall': 0.0523,
        'f1': 0.0896, 'auc': 0.827, 'confusion_matrix': [[11962, 33], [272, 15]]
    },
    'SVM': {
        'accuracy': 0.9765, 'precision': 0.3333, 'recall': 0.0070,
        'f1': 0.0137, 'auc': 0.590, 'confusion_matrix': [[11991, 4], [285, 2]]
    },
    'KNN': {
        'accuracy': 0.9589, 'precision': 0.1414, 'recall': 0.1498,
        'f1': 0.1455, 'auc': 0.695, 'confusion_matrix': [[11734, 261], [244, 43]]
    }
}

# Feature importance values
feature_importances = {
    'Decision Tree': {
        'Retained Earnings': 0.072059, 'Market Value': 0.072055,
        'Inventory': 0.070231, 'D&A': 0.068246, 'Gross Profit': 0.067548,
        'Total Receivables': 0.065696, 'Current Assets': 0.065387,
        'Total Long-term Debt': 0.064578, 'Total Assets': 0.056883,
        'Total Current Liabilities': 0.055932, 'Net Income': 0.055526,
        'Total Liabilities': 0.052951, 'Cost of Goods Sold': 0.051296,
        'Total Operating Expenses': 0.047349, 'EBITDA': 0.041733,
        'EBIT': 0.041661, 'Total Revenue': 0.027468, 'Net Sales': 0.023400
    },
    'Gradient Boosting': {
        'Total Long-term Debt': 0.115407, 'Net Income': 0.113170,
        'Retained Earnings': 0.088011, 'Market Value': 0.083996,
        'Inventory': 0.075858, 'Total Operating Expenses': 0.071508,
        'Current Assets': 0.068556, 'Total Receivables': 0.066965,
        'Gross Profit': 0.056605, 'D&A': 0.045299, 'Total Liabilities': 0.040103,
        'EBITDA': 0.031667, 'EBIT': 0.030457, 'Net Sales': 0.028807,
        'Cost of Goods Sold': 0.028534, 'Total Current Liabilities': 0.022211,
        'Total Assets': 0.017786, 'Total Revenue': 0.015061
    },
    'Random Forest': {
        'Retained Earnings': 0.065674, 'Market Value': 0.062897,
        'D&A': 0.061341, 'Current Assets': 0.059910,
        'Total Receivables': 0.059713, 'Gross Profit': 0.058533,
        'Total Liabilities': 0.057575, 'Total Assets': 0.057426,
        'Total Current Liabilities': 0.055479, 'Inventory': 0.054929,
        'Total Long-term Debt': 0.054677, 'Net Income': 0.053633,
        'Cost of Goods Sold': 0.053133, 'EBITDA': 0.051601,
        'EBIT': 0.050618, 'Total Operating Expenses': 0.049919,
        'Total Revenue': 0.046852, 'Net Sales': 0.046092
    },
    'Logistic Regression': {
        'Market Value': 1.102307, 'Current Assets': 0.976875,
        'Total Current Liabilities': 0.500875, 'EBIT': 0.418057,
        'Total Long-term Debt': 0.366918, 'Total Liabilities': 0.335098,
        'EBITDA': 0.309482, 'Inventory': 0.285948, 'Total Assets': 0.231877,
        'Gross Profit': 0.153693, 'Cost of Goods Sold': 0.065107,
        'Total Operating Expenses': 0.056967, 'Retained Earnings': 0.054134,
        'Total Receivables': 0.040750, 'Net Income': 0.019487,
        'D&A': 0.006214, 'Net Sales': 0.001644, 'Total Revenue': 0.001644
    },
    'KNN': {
        'Inventory': 0.048982, 'D&A': 0.048754,
        'Total Long-term Debt': 0.042688, 'Gross Profit': 0.039603,
        'Retained Earnings': 0.030695, 'Total Liabilities': 0.023482,
        'Cost of Goods Sold': 0.005708, 'EBIT': 0.004975,
        'Total Operating Expenses': 0.001930, 'Total Revenue': 0.001262,
        'Net Sales': 0.001262, 'Total Current Liabilities': 0.000244,
        'Current Assets': -0.000090, 'Total Receivables': -0.000627,
        'Total Assets': -0.001449, 'EBITDA': -0.001767,
        'Market Value': -0.002597, 'Net Income': -0.004633
    },
    'SVM': {
        'Current Assets': 0.000147, 'Total Receivables': 0.000090,
        'Gross Profit': 0.000008, 'Total Revenue': 0.000000,
        'Cost of Goods Sold': 0.000000, 'Net Sales': 0.000000,
        'Total Assets': 0.000000, 'EBITDA': 0.000000,
        'D&A': 0.000000, 'Total Operating Expenses': 0.000000,
        'Market Value': -0.000008, 'Inventory': -0.000008,
        'Total Current Liabilities': -0.000008, 'Net Income': -0.000016,
        'EBIT': -0.000016, 'Total Long-term Debt': -0.000090,
        'Total Liabilities': -0.000163, 'Retained Earnings': -0.000220
    }
}

# ROC curve data
roc_curves = {
    'Decision Tree': {
        'fpr': [0.0,0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        'tpr': [0.0,0.05,0.1,0.15,0.2,0.24,0.32,0.40,0.5,0.6,0.7,0.8,0.9,1.0],
        'auc': 0.574
    },
    'Gradient Boosting': {
        'fpr': [0.0,0.001,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.4,0.6,0.8,1.0],
        'tpr': [0.0,0.1,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.9,0.95,0.98,1.0],
        'auc': 0.827
    },
    'Random Forest': {
        'fpr': [0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.4,0.6,0.8,1.0],
        'tpr': [0.0,0.08,0.2,0.3,0.4,0.5,0.65,0.75,0.85,0.92,0.98,1.0],
        'auc': 0.835
    },
    'Logistic Regression': {
        'fpr': [0.0,0.002,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.7,0.9,1.0],
        'tpr': [0.0,0.1,0.2,0.3,0.4,0.5,0.65,0.75,0.85,0.92,0.98,1.0],
        'auc': 0.827
    },
    'SVM': {
        'fpr': [0.0,0.0003,0.001,0.005,0.01,0.05,0.1,0.3,0.5,0.7,0.9,1.0],
        'tpr': [0.0,0.05,0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,0.95,1.0],
        'auc': 0.590
    },
    'KNN': {
        'fpr': [0.0,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0],
        'tpr': [0.0,0.05,0.1,0.15,0.2,0.25,0.35,0.45,0.6,0.75,0.9,1.0],
        'auc': 0.695
    }
}

# Z-Score calculation
def calculate_zscore(df):
    """Calculate Altman Z-Score and classifications."""
    try:
        z = pd.DataFrame(index=df.index)
        z['T1'] = (df['Current Assets'] - df['Total Current Liabilities']) / df['Total Assets']
        z['T2'] = df['Retained Earnings'] / df['Total Assets']
        z['T3'] = df['EBIT'] / df['Total Assets']
        z['T4'] = df['Market Value'] / df['Total Liabilities']
        z['T5'] = df['Net Sales'] / df['Total Assets']
        z['Z-Score'] = (1.2*z['T1'] + 1.4*z['T2'] + 3.3*z['T3'] + 0.6*z['T4'] + 0.99*z['T5'])
        z.replace([np.inf, -np.inf], np.nan, inplace=True)
        if z['Z-Score'].isna().any():
            z['Z-Score'].fillna(z['Z-Score'].mean(), inplace=True)
        thresholds = (1.8, 2.99)
        z['Z-Score Status'] = pd.cut(
            z['Z-Score'],
            bins=[-np.inf, thresholds[0], thresholds[1], np.inf],
            labels=['Distress', 'Grey', 'Safe']
        )
        z['Z-Score Prediction'] = (z['Z-Score Status'] == 'Distress').astype(int)
        return z
    except Exception:
        return pd.DataFrame()

# Sidebar navigation
pages = [
    "Overview",
    "Data Overview",
    "Model Comparison",
    "ROC Curves",
    "Feature Importance",
    "Confusion Matrices",
    "Z-Score Analysis"
]
selected_page = st.sidebar.radio("", pages, key="sidebar_nav")
st.sidebar.markdown("---")
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# --- Overview Page ---
if selected_page == "Overview":
    st.markdown('<p class="main-header">Bankruptcy Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard presents a comprehensive analysis of bankruptcy prediction models using financial data from American companies.
    """)
    st.markdown('<p class="sub-header">Overview</p>', unsafe_allow_html=True)
    st.markdown("""
    ### Project Summary
    - Dataset: Kaggle American Companies Bankruptcy Prediction (1999–2018)
    - Features: 18 financial indicators
    - Target: Bankrupt vs Alive
    - Models: Decision Tree, Gradient Boosting, Random Forest, Logistic Regression, SVM, KNN
    """)
    metrics_df = pd.DataFrame({
        'Accuracy': [metrics[m]['accuracy'] for m in metrics],
        'Precision': [metrics[m]['precision'] for m in metrics],
        'Recall': [metrics[m]['recall'] for m in metrics],
        'F1 Score': [metrics[m]['f1'] for m in metrics],
        'AUC': [metrics[m]['auc'] for m in metrics]
    }, index=metrics.keys())
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best AUC", f"{metrics_df['AUC'].max():.3f}", metrics_df['AUC'].idxmax())
    with col2:
        st.metric("Best F1", f"{metrics_df['F1 Score'].max():.3f}", metrics_df['F1 Score'].idxmax())
    with col3:
        st.metric("Best Recall", f"{metrics_df['Recall'].max():.3f}", metrics_df['Recall'].idxmax())
    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.bar(metrics_df.index, metrics_df['AUC'], color='#395c40')
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.3f}", ha='center')
    ax.set_ylabel("AUC"); ax.set_title("Model AUC Comparison"); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    st.pyplot(fig)
    if st.session_state["data_loaded"]:
        st.markdown("### Dataset Preview")
        preview = data.copy()
        cols = []
        for c in ['status_label','Bankrupt','year']:
            if c in preview.columns: cols.append(c)
        for c in ['Current Assets','Total Assets','Net Income','EBIT','Market Value']:
            if c in preview.columns: cols.append(c)
        if not cols: cols = preview.columns[:5].tolist()
        st.dataframe(preview[cols].head())

# --- Data Overview Page ---
elif selected_page == "Data Overview":
    st.markdown('<p class="page-header">Data Overview</p>', unsafe_allow_html=True)
    st.markdown("""
    **Dataset source:** https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset
    """)
    raw_path = Path("data/american_bankruptcy.csv")
    if not raw_path.exists():
        st.error("Could not load raw data. Ensure 'data/american_bankruptcy.csv' exists.")
    else:
        raw_df = pd.read_csv(raw_path)
        st.markdown("### Original Column Names")
        st.write(raw_df.columns.tolist())
        st.markdown("### First 15 Rows (Original Data)")
        st.dataframe(raw_df.head(15))
        st.markdown("### Column Mapping Applied")
        st.write(rename_map)
        mapped_df = raw_df.rename(columns=rename_map)
        st.markdown("### First 15 Rows (Mapped Data)")
        st.dataframe(mapped_df.head(15))
        if mapped_df.isna().sum().sum() == 0:
            st.markdown("**No missing values detected; the data was inherently clean.**")

# --- Model Comparison Page ---
elif selected_page == "Model Comparison":
    st.markdown('<p class="page-header">Model Performance Comparison</p>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        'Accuracy': [metrics[m]['accuracy'] for m in metrics],
        'Precision': [metrics[m]['precision'] for m in metrics],
        'Recall': [metrics[m]['recall'] for m in metrics],
        'F1 Score': [metrics[m]['f1'] for m in metrics],
        'AUC': [metrics[m]['auc'] for m in metrics]
    }, index=metrics.keys())
    st.markdown("### Performance Metrics")
    st.dataframe(metrics_df.style.highlight_max(axis=0))
    st.markdown("### Metric Comparison")
    opts = ["Accuracy","Precision","Recall","F1 Score","AUC"]
    chosen = st.multiselect("Select metrics to compare", opts, default=["Recall","F1 Score","AUC"])
    if chosen:
        fig, axes = plt.subplots(1, len(chosen), figsize=(5*len(chosen),5))
        if len(chosen)==1: axes=[axes]
        for ax, metric in zip(axes, chosen):
            df_sorted = metrics_df.sort_values(metric, ascending=False)
            bars = ax.bar(df_sorted.index, df_sorted[metric], color='#395c40')
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.3f}", ha='center')
            ax.set_title(metric); ax.tick_params(axis='x', rotation=45)
        plt.tight_layout(); st.pyplot(fig)

# --- ROC Curves Page ---
elif selected_page == "ROC Curves":
    st.markdown('<p class="page-header">ROC Curve Analysis</p>', unsafe_allow_html=True)
    st.markdown("""
    ROC curves plot True Positive Rate vs False Positive Rate at different thresholds.
    """)
    models = list(metrics.keys())
    sel = st.multiselect("Select models", models, default=models[:3])
    if sel:
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot([0,1],[0,1],'--', color='gray', alpha=0.8, label='Random')
        colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
        for i, m in enumerate(sel):
            ax.plot(roc_curves[m]['fpr'], roc_curves[m]['tpr'], lw=2,
                    color=colors[i%len(colors)], label=f"{m} (AUC={roc_curves[m]['auc']:.3f})")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves"); ax.legend(loc='lower right'); ax.grid(alpha=0.3)
        st.pyplot(fig)

# --- Feature Importance Page ---
elif selected_page == "Feature Importance":
    st.markdown('<p class="page-header">Feature Importance Analysis</p>', unsafe_allow_html=True)
    models = list(feature_importances.keys())
    sel = st.selectbox("Select model", models)
    imp = pd.Series(feature_importances[sel]).sort_values(ascending=False)
    n = st.slider("Number of top features to display", 5, len(imp), 10)
    st.markdown(f"### Top {n} Features for {sel}")
    fig, ax = plt.subplots(figsize=(10,8))
    top = imp.head(n)
    bars = ax.barh(top.index[::-1], top.values[::-1], color='#395c40')
    maxv = imp.max()
    for bar in bars:
        ax.text(bar.get_width()+maxv*0.05, bar.get_y()+bar.get_height()/2,
                f"{bar.get_width():.3f}", va='center', ha='left')
    ax.set_xlabel("Importance"); ax.set_title(f"{sel} Feature Importance")
    plt.tight_layout(); st.pyplot(fig)

# --- Confusion Matrices Page ---
elif selected_page == "Confusion Matrices":
    st.markdown('<p class="page-header">Confusion Matrix Analysis</p>', unsafe_allow_html=True)
    models = list(metrics.keys())
    sel = st.selectbox("Select model", models)
    tn, fp = metrics[sel]['confusion_matrix'][0]
    fn, tp = metrics[sel]['confusion_matrix'][1]
    total = tn+fp+fn+tp
    acc = (tn+tp)/total
    prec = tp/(tp+fp) if tp+fp>0 else 0
    rec = tp/(tp+fn) if tp+fn>0 else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
    st.markdown(f"### {sel} Confusion Matrix")
    col1, col2 = st.columns([2,1])
    with col1:
        cm_df = pd.DataFrame([[tn,fp],[fn,tp]],
                             index=["Actual Healthy","Actual Bankrupt"],
                             columns=["Predicted Healthy","Predicted Bankrupt"])
        st.dataframe(cm_df)
        pct = np.array([[100*tn/(tn+fp),100*fp/(tn+fp)],
                        [100*fn/(fn+tp),100*tp/(fn+tp)]])
        html = f"""
        <style>
        .cm-box {{padding:20px;text-align:center;margin:5px;font-weight:bold;color:white;}}
        .box-container {{display:grid;grid-template:1fr 1fr / 1fr 1fr;gap:10px;margin:20px 0;}}
        .tn {{background-color:rgba(57,92,64,0.8);}}
        .fp,.fn {{background-color:rgba(166,54,3,0.8);}}
        .tp {{background-color:rgba(57,92,64,0.8);}}
        </style>
        <div class="box-container">
          <div class="cm-box tn">TN<br>{tn}<br>({pct[0,0]:.1f}%)</div>
          <div class="cm-box fp">FP<br>{fp}<br>({pct[0,1]:.1f}%)</div>
          <div class="cm-box fn">FN<br>{fn}<br>({pct[1,0]:.1f}%)</div>
          <div class="cm-box tp">TP<br>{tp}<br>({pct[1,1]:.1f}%)</div>
        </div>"""
        st.markdown(html, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        - **Accuracy**: {acc:.4f}  
        - **Precision**: {prec:.4f}  
        - **Recall**: {rec:.4f}  
        - **F1 Score**: {f1:.4f}
        """)

# --- Z-Score Analysis Page ---
elif selected_page == "Z-Score Analysis":
    st.markdown('<p class="page-header">Altman Z-Score Analysis</p>', unsafe_allow_html=True)
    st.markdown("""
    Z = 1.2·T1 + 1.4·T2 + 3.3·T3 + 0.6·T4 + 0.99·T5  
    where:
    - T1 = (Current Assets – Current Liabilities) / Total Assets  
    - T2 = Retained Earnings / Total Assets  
    - T3 = EBIT / Total Assets  
    - T4 = Market Value / Total Liabilities  
    - T5 = Net Sales / Total Assets  
    """)
    if st.session_state["data_loaded"]:
        req = ['Current Assets','Total Current Liabilities','Retained Earnings',
               'Total Assets','EBIT','Market Value','Total Liabilities','Net Sales']
        missing = [c for c in req if c not in data.columns]
        if missing:
            st.error(f"Missing columns for Z-Score: {missing}")
        else:
            zdf = calculate_zscore(data)
            with st.expander("Z-Score Statistics"):
                st.write(zdf['Z-Score'].describe())
                st.write(zdf['Z-Score Status'].value_counts())
            if 'Bankrupt' in data.columns:
                actual = data['Bankrupt']; pred = zdf['Z-Score Prediction']
                acc_z = (pred==actual).mean()
                prec_z = (pred & actual).sum()/pred.sum() if pred.sum()>0 else 0
                rec_z = (pred & actual).sum()/actual.sum() if actual.sum()>0 else 0
                f1_z = 2*prec_z*rec_z/(prec_z+rec_z) if prec_z+rec_z>0 else 0
                st.markdown("### Z-Score vs ML Models")
                cmp = pd.DataFrame({
                    'Model': ['Altman Z-Score']+list(metrics.keys()),
                    'Accuracy': [acc_z]+[metrics[m]['accuracy'] for m in metrics],
                    'Precision': [prec_z]+[metrics[m]['precision'] for m in metrics],
                    'Recall': [rec_z]+[metrics[m]['recall'] for m in metrics],
                    'F1 Score': [f1_z]+[metrics[m]['f1'] for m in metrics]
                }).set_index('Model')
                st.dataframe(cmp.style.highlight_max(axis=0))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:0.8em;">
Bankruptcy Prediction Dashboard | Created with Streamlit
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
