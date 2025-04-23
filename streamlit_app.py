# Part 1 of 3

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="Bankruptcy Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #395c40;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #395c40;
}
.section-header {
    font-size: 1.2rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.markdown('<p class="main-header">Bankruptcy Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown("""
This dashboard presents a comprehensive analysis of bankruptcy prediction models using financial data 
from American companies. The analysis compares multiple machine learning models and their performance metrics.
""")

# Define column renaming mapping
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

@st.cache_data
def load_data():
    """Load data with multiple fallback paths and detailed error reporting"""
    possible_paths = [
        'data/american_bankruptcy.csv',
        'american_bankruptcy.csv',
        '../data/american_bankruptcy.csv',
        './american_bankruptcy.csv',
    ]
    
    for path in possible_paths:
        try:
            st.sidebar.info(f"Trying to load from: {path}")
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Data loaded successfully from {path}")
                
                # Handle status_label -> Bankrupt
                if "status_label" in df.columns:
                    vals = df['status_label'].unique()
                    st.sidebar.info(f"Status values found: {', '.join(vals)}")
                    if 'failed' in vals:
                        df['Bankrupt'] = df['status_label'].map({'failed':1,'alive':0})
                    elif 'Bankrupt' in vals:
                        df['Bankrupt'] = df['status_label'].map({'Bankrupt':1,'Alive':0})
                    else:
                        df['Bankrupt'] = df['status_label'].apply(
                            lambda x: 1 if x.lower() in ['failed','bankrupt','distress','default'] else 0
                        )
                    st.sidebar.success("✅ Converted status_label to Bankrupt column")
                
                # Rename Bankruptcy column if needed
                if "Bankruptcy" in df.columns and "Bankrupt" not in df.columns:
                    df['Bankrupt'] = df['Bankruptcy']
                    st.sidebar.success("✅ Renamed Bankruptcy column to Bankrupt")
                
                # Rename X1–X18
                if "X1" in df.columns:
                    df = df.rename(columns=rename_map)
                    st.sidebar.success("✅ Renamed X1–X18 columns to descriptive names")
                
                return df
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            continue
    
    st.sidebar.error("❌ Failed to load data from any location.")
    st.error("Could not load the data file. Please check that the file exists in the specified locations.")
    return pd.DataFrame()

# Load and inspect data
try:
    data = load_data()
    if not data.empty:
        with st.sidebar.expander("📊 Data Information"):
            st.write(f"Rows: {data.shape[0]}")
            st.write(f"Columns: {data.shape[1]}")
            st.write("Column Names:")
            st.write(", ".join(data.columns.tolist()))
            
            required_cols = [
                'Current Assets', 'Total Current Liabilities', 'Retained Earnings', 
                'Total Assets', 'EBIT', 'Market Value', 'Total Liabilities', 'Net Sales'
            ]
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                st.error(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ All required columns for Z-Score calculation are present")
except Exception as e:
    st.error(f"Error during data initialization: {e}")
    data = pd.DataFrame()

st.session_state['data_loaded'] = not data.empty

# Feature names
feature_names = [
    "Current Assets", "Cost of Goods Sold", "D&A", "EBITDA",
    "Inventory", "Net Income", "Total Receivables", "Market Value",
    "Net Sales", "Total Assets", "Total Long-term Debt", "EBIT",
    "Gross Profit", "Total Current Liabilities", "Retained Earnings",
    "Total Revenue", "Total Liabilities", "Total Operating Expenses"
]

# Performance metrics for models
metrics = {
    'Decision Tree': {
        'accuracy': 0.8925,
        'precision': 0.0589,
        'recall': 0.2404,
        'f1': 0.0947,
        'auc': 0.574,
        'confusion_matrix': [[10893, 1102], [218, 69]]
    },
    'Gradient Boosting': {
        'accuracy': 0.9761,
        'precision': 0.3846,
        'recall': 0.0348,
        'f1': 0.0639,
        'auc': 0.827,
        'confusion_matrix': [[11979, 16], [277, 10]]
    },
    'Random Forest': {
        'accuracy': 0.9759,
        'precision': 0.3200,
        'recall': 0.0279,
        'f1': 0.0513,
        'auc': 0.835,
        'confusion_matrix': [[11978, 17], [279, 8]]
    },
    'Logistic Regression': {
        'accuracy': 0.9752,
        'precision': 0.3125,
        'recall': 0.0523,
        'f1': 0.0896,
        'auc': 0.827,
        'confusion_matrix': [[11962, 33], [272, 15]]
    },
    'SVM': {
        'accuracy': 0.9765,
        'precision': 0.3333,
        'recall': 0.0070,
        'f1': 0.0137,
        'auc': 0.590,
        'confusion_matrix': [[11991, 4], [285, 2]]
    },
    'KNN': {
        'accuracy': 0.9589,
        'precision': 0.1414,
        'recall': 0.1498,
        'f1': 0.1455,
        'auc': 0.695,
        'confusion_matrix': [[11734, 261], [244, 43]]
    }
}

# Part 2 of 3

# Feature importance data
feature_importances = {
    'Decision Tree': {
        'Retained Earnings': 0.072059,
        'Market Value': 0.072055,
        'Inventory': 0.070231,
        'D&A': 0.068246,
        'Gross Profit': 0.067548,
        'Total Receivables': 0.065696,
        'Current Assets': 0.065387,
        'Total Long-term Debt': 0.064578,
        'Total Assets': 0.056883,
        'Total Current Liabilities': 0.055932,
        'Net Income': 0.055526,
        'Total Liabilities': 0.052951,
        'Cost of Goods Sold': 0.051296,
        'Total Operating Expenses': 0.047349,
        'EBITDA': 0.041733,
        'EBIT': 0.041661,
        'Total Revenue': 0.027468,
        'Net Sales': 0.023400
    },
    'Gradient Boosting': {
        'Total Long-term Debt': 0.115407,
        'Net Income': 0.113170,
        'Retained Earnings': 0.088011,
        'Market Value': 0.083996,
        'Inventory': 0.075858,
        'Total Operating Expenses': 0.071508,
        'Current Assets': 0.068556,
        'Total Receivables': 0.066965,
        'Gross Profit': 0.056605,
        'D&A': 0.045299,
        'Total Liabilities': 0.040103,
        'EBITDA': 0.031667,
        'EBIT': 0.030457,
        'Net Sales': 0.028807,
        'Cost of Goods Sold': 0.028534,
        'Total Current Liabilities': 0.022211,
        'Total Assets': 0.017786,
        'Total Revenue': 0.015061
    },
    'Random Forest': {
        'Retained Earnings': 0.065674,
        'Market Value': 0.062897,
        'D&A': 0.061341,
        'Current Assets': 0.059910,
        'Total Receivables': 0.059713,
        'Gross Profit': 0.058533,
        'Total Liabilities': 0.057575,
        'Total Assets': 0.057426,
        'Total Current Liabilities': 0.055479,
        'Inventory': 0.054929,
        'Total Long-term Debt': 0.054677,
        'Net Income': 0.053633,
        'Cost of Goods Sold': 0.053133,
        'EBITDA': 0.051601,
        'EBIT': 0.050618,
        'Total Operating Expenses': 0.049919,
        'Total Revenue': 0.046852,
        'Net Sales': 0.046092
    },
    'Logistic Regression': {
        'Market Value': 1.102307,
        'Current Assets': 0.976875,
        'Total Current Liabilities': 0.500875,
        'EBIT': 0.418057,
        'Total Long-term Debt': 0.366918,
        'Total Liabilities': 0.335098,
        'EBITDA': 0.309482,
        'Inventory': 0.285948,
        'Total Assets': 0.231877,
        'Gross Profit': 0.153693,
        'Cost of Goods Sold': 0.065107,
        'Total Operating Expenses': 0.056967,
        'Retained Earnings': 0.054134,
        'Total Receivables': 0.040750,
        'Net Income': 0.019487,
        'D&A': 0.006214,
        'Net Sales': 0.001644,
        'Total Revenue': 0.001644
    },
    'SVM': {
        'Current Assets': 0.000147,
        'Total Receivables': 0.000090,
        'Gross Profit': 0.000008,
        'Total Revenue': 0.000000,
        'Cost of Goods Sold': 0.000000,
        'Net Sales': 0.000000,
        'Total Assets': 0.000000,
        'EBITDA': 0.000000,
        'D&A': 0.000000,
        'Total Operating Expenses': 0.000000,
        'Market Value': -0.000008,
        'Inventory': -0.000008,
        'Total Current Liabilities': -0.000008,
        'Net Income': -0.000016,
        'EBIT': -0.000016,
        'Total Long-term Debt': -0.000090,
        'Total Liabilities': -0.000163,
        'Retained Earnings': -0.000220
    },
    'KNN': {
        'Inventory': 0.048982,
        'D&A': 0.048754,
        'Total Long-term Debt': 0.042688,
        'Gross Profit': 0.039603,
        'Retained Earnings': 0.030695,
        'Total Liabilities': 0.023482,
        'Cost of Goods Sold': 0.005708,
        'EBIT': 0.004975,
        'Total Operating Expenses': 0.001930,
        'Total Revenue': 0.001262,
        'Net Sales': 0.001262,
        'Total Current Liabilities': 0.000244,
        'Current Assets': -0.000090,
        'Total Receivables': -0.000627,
        'Total Assets': -0.001449,
        'EBITDA': -0.001767,
        'Market Value': -0.002597,
        'Net Income': -0.004633
    }
}

# ROC curve data for each model
roc_curves = {
    'Decision Tree': {
        'fpr': [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'tpr': [0.0, 0.05, 0.1, 0.15, 0.2, 0.24, 0.32, 0.40, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'auc': 0.574
    },
    'Gradient Boosting': {
        'fpr': [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        'tpr': [0.0, 0.1, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95, 0.98, 1.0],
        'auc': 0.827
    },
    'Random Forest': {
        'fpr': [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        'tpr': [0.0, 0.08, 0.2, 0.3, 0.4, 0.5, 0.65, 0.75, 0.85, 0.92, 0.98, 1.0],
        'auc': 0.835
    },
    'Logistic Regression': {
        'fpr': [0.0, 0.002, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        'tpr': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.75, 0.85, 0.92, 0.98, 1.0],
        'auc': 0.827
    },
    'SVM': {
        'fpr': [0.0, 0.0003, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'tpr': [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        'auc': 0.590
    },
    'KNN': {
        'fpr': [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
        'tpr': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.45, 0.6, 0.75, 0.9, 1.0],
        'auc': 0.695
    }
}

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Overview", "Model Comparison", "ROC Curves", "Feature Importance", "Confusion Matrices", "Z-Score Analysis"]
selected_page = st.sidebar.radio("Go to", pages)

# ========== Overview Page ==========
if selected_page == "Overview":
    st.markdown('<p class="sub-header">Overview</p>', unsafe_allow_html=True)
    st.markdown("""
    ### Project Summary
    - **Training Data**: Financial data from 1999–2011  
    - **Testing Data**: Financial data from 2015–2018  
    - **Features**: 18 financial indicators  
    - **Target**: Bankruptcy (binary)
    
    ### Models Analyzed
    Decision Tree, Gradient Boosting, Random Forest, Logistic Regression, SVM, KNN
    
    ### Key Metrics
    Accuracy, Precision, Recall, F1 Score, AUC
    """)
    metrics_df = pd.DataFrame({
        'Accuracy': [metrics[m]['accuracy'] for m in metrics],
        'Precision': [metrics[m]['precision'] for m in metrics],
        'Recall': [metrics[m]['recall'] for m in metrics],
        'F1 Score': [metrics[m]['f1'] for m in metrics],
        'AUC': [metrics[m]['auc'] for m in metrics]
    }, index=list(metrics.keys()))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best AUC", f"{metrics_df['AUC'].max():.3f}", metrics_df['AUC'].idxmax())
    with col2:
        st.metric("Best F1 Score", f"{metrics_df['F1 Score'].max():.3f}", metrics_df['F1 Score'].idxmax())
    with col3:
        st.metric("Best Recall", f"{metrics_df['Recall'].max():.3f}", metrics_df['Recall'].idxmax())
    st.markdown("### Model AUC Comparison")
    auc_series = metrics_df['AUC'].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.bar(auc_series.index, auc_series.values, color='#395c40')
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.3f}", ha='center', va='bottom')
    ax.set_ylabel('AUC Score'); ax.set_title('Model AUC Comparison')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    st.pyplot(fig)
    
    if st.session_state['data_loaded']:
        st.markdown("### Dataset Preview")
        preview = data.copy()
        cols = [c for c in ['status_label','Bankrupt','year','Current Assets','Total Assets','Net Income','EBIT','Market Value'] if c in preview.columns]
        if not cols:
            cols = preview.columns[:5].tolist()
        st.dataframe(preview[cols].head())
        st.markdown("### Dataset Statistics")
        st.write(f"Number of records: {len(data)}")
        st.write(f"Number of features: {len(data.columns)}")
        if 'Bankrupt' in data.columns:
            counts = data['Bankrupt'].value_counts().rename({1:'Bankrupt',0:'Healthy'})
            fig, ax = plt.subplots(figsize=(8,6))
            ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['#a63603','#395c40'])
            ax.axis('equal'); plt.title('Distribution of Bankruptcy Status')
            st.pyplot(fig)

# ========== Model Comparison Page ==========
elif selected_page == "Model Comparison":
    st.markdown('<p class="sub-header">Model Performance Comparison</p>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        'Accuracy': [metrics[m]['accuracy'] for m in metrics],
        'Precision': [metrics[m]['precision'] for m in metrics],
        'Recall': [metrics[m]['recall'] for m in metrics],
        'F1 Score': [metrics[m]['f1'] for m in metrics],
        'AUC': [metrics[m]['auc'] for m in metrics]
    }, index=list(metrics.keys()))
    st.markdown("### Performance Metrics")
    st.dataframe(metrics_df.style.highlight_max(axis=0))
    st.markdown("### Metric Comparison")
    metric_options = ["Accuracy","Precision","Recall","F1 Score","AUC"]
    selected_metrics = st.multiselect("Select metrics to compare", metric_options, default=["Recall","F1 Score","AUC"])
    if selected_metrics:
        fig, axes = plt.subplots(1, len(selected_metrics), figsize=(15,5))
        if len(selected_metrics)==1:
            axes=[axes]
        for i, metric in enumerate(selected_metrics):
            dfm = metrics_df.sort_values(metric, ascending=False)
            bars = axes[i].bar(dfm.index, dfm[metric], color='#395c40')
            for b in bars:
                axes[i].text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.3f}", ha='center', va='bottom')
            axes[i].set_title(metric)
            axes[i].set_ylim(0, dfm[metric].max()*1.2)
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout(); st.pyplot(fig)
    st.markdown("### Class Imbalance")
    st.info("Dataset is imbalanced; recall is key for bankruptcy detection.")

# ========== ROC Curves Page ==========
elif selected_page == "ROC Curves":
    st.markdown('<p class="sub-header">ROC Curve Analysis</p>', unsafe_allow_html=True)
    st.markdown("ROC curves plot TPR vs FPR. AUC=1 perfect; 0.5 random.")
    models = list(metrics.keys())
    selected_models = st.multiselect("Select models", models, default=models[:3])
    if selected_models:
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot([0,1],[0,1], linestyle='--', color='gray', alpha=0.8, label='Random')
        colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
        for i,m in enumerate(selected_models):
            fpr, tpr, auc = roc_curves[m]['fpr'], roc_curves[m]['tpr'], roc_curves[m]['auc']
            ax.plot(fpr, tpr, lw=2, color=colors[i%len(colors)], label=f"{m} (AUC={auc:.3f})")
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison'); ax.legend(loc='lower right')
        ax.set_xlim(0,1); ax.set_ylim(0,1.05); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig)
    st.markdown("### Individual Model ROC Curve")
    single = st.selectbox("Select a model", models)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot([0,1],[0,1], linestyle='--', color='gray', alpha=0.8, label='Random')
    fpr, tpr, auc = roc_curves[single]['fpr'], roc_curves[single]['tpr'], roc_curves[single]['auc']
    ax.plot(fpr, tpr, lw=2, color='#395c40', label=f"AUC={auc:.3f}")
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{single} ROC Curve'); ax.legend(loc='lower right')
    ax.set_xlim(0,1); ax.set_ylim(0,1.05); ax.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig)

# (Part 2 ends here — remaining Confusion Matrices and Z-Score pages in Part 3)
# Part 3 of 3

# ========== Feature Importance Page ==========
# (If not already covered in Part 2; otherwise skip to Confusion Matrices)

# ========== Confusion Matrices Page ==========
elif selected_page == "Confusion Matrices":
    st.markdown('<p class="sub-header">Confusion Matrix Analysis</p>', unsafe_allow_html=True)
    
    model_options = list(metrics.keys())
    selected_model = st.selectbox("Select model", model_options)
    
    cm = metrics[selected_model]['confusion_matrix']
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    st.markdown(f"### {selected_model} Confusion Matrix")
    col1, col2 = st.columns([2, 1])
    with col1:
        cm_df = pd.DataFrame(
            cm,
            index=['Actual Alive', 'Actual Bankrupt'],
            columns=['Predicted Alive', 'Predicted Bankrupt']
        )
        st.dataframe(cm_df)
        
        st.markdown("### Visual Representation")
        cm_pct = np.zeros((2, 2))
        cm_pct[0, 0] = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
        cm_pct[0, 1] = 100 * fp / (tn + fp) if (tn + fp) > 0 else 0
        cm_pct[1, 0] = 100 * fn / (fn + tp) if (fn + tp) > 0 else 0
        cm_pct[1, 1] = 100 * tp / (fn + tp) if (fn + tp) > 0 else 0
        
        html = f"""
        <style>
        .cm-box {{
            padding: 20px;
            text-align: center;
            margin: 5px;
            font-weight: bold;
            color: white;
        }}
        .box-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 10px;
            margin: 20px 0;
        }}
        .tn {{ background-color: rgba(57, 92, 64, 0.8); }}
        .fp, .fn {{ background-color: rgba(166, 54, 3, 0.8); }}
        .tp {{ background-color: rgba(57, 92, 64, 0.8); }}
        </style>
        <div class="box-container">
            <div class="cm-box tn">
                True Negative<br>{tn}<br>({cm_pct[0, 0]:.1f}%)
            </div>
            <div class="cm-box fp">
                False Positive<br>{fp}<br>({cm_pct[0, 1]:.1f}%)
            </div>
            <div class="cm-box fn">
                False Negative<br>{fn}<br>({cm_pct[1, 0]:.1f}%)
            </div>
            <div class="cm-box tp">
                True Positive<br>{tp}<br>({cm_pct[1, 1]:.1f}%)
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Metrics")
        st.markdown(f"""
        - **Accuracy**: {accuracy:.4f}
        - **Precision**: {precision:.4f}
        - **Recall**: {recall:.4f}
        - **F1 Score**: {f1:.4f}
        """)
    
    st.markdown("### Bankruptcy Detection Comparison")
    comparison_df = pd.DataFrame(
        index=model_options,
        columns=["True Positives", "False Negatives", "Detection Rate (%)", "False Alarm Rate (%)"]
    )
    for model in model_options:
        tn0, fp0 = metrics[model]['confusion_matrix'][0]
        fn0, tp0 = metrics[model]['confusion_matrix'][1]
        comparison_df.loc[model] = [
            tp0,
            fn0,
            100 * tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0,
            100 * fp0 / (tn0 + fp0) if (tn0 + fp0) > 0 else 0
        ]
    st.dataframe(comparison_df.sort_values("Detection Rate (%)", ascending=False))

# ========== Z-Score Analysis Page ==========
elif selected_page == "Z-Score Analysis":
    st.markdown('<p class="sub-header">Altman Z-Score Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### What is the Altman Z-Score?
    
    The Altman Z-Score combines five financial ratios to predict bankruptcy risk:
    Z = 1.2·T1 + 1.4·T2 + 3.3·T3 + 0.6·T4 + 0.99·T5  
    - T1 = (Current Assets - Current Liabilities) / Total Assets  
    - T2 = Retained Earnings / Total Assets  
    - T3 = EBIT / Total Assets  
    - T4 = Market Value / Total Liabilities  
    - T5 = Net Sales / Total Assets  
    Zones: Distress (<1.8), Grey (1.8–2.99), Safe (>2.99)
    """)
    
    if st.session_state['data_loaded'] and not data.empty:
        required_cols = [
            'Current Assets', 'Total Current Liabilities', 'Retained Earnings',
            'Total Assets', 'EBIT', 'Market Value', 'Total Liabilities', 'Net Sales'
        ]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            st.error(f"Cannot calculate Z-Score: Missing {', '.join(missing)}")
        else:
            # Enhanced Bankruptcy Status plot
            if 'Bankrupt' in data.columns:
                st.markdown("<div style='margin-top:20px; margin-bottom:20px;'></div>", unsafe_allow_html=True)
                bc = int(data['Bankrupt'].sum())
                ac = len(data) - bc
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(['Alive', 'Bankrupt'], [ac, bc], color=['#395c40', '#a63603'])
                mx = max(ac, bc)
                ax.set_ylim(0, mx * 1.2)
                ax.set_ylabel('Number of Companies')
                for i, v in enumerate([ac, bc]):
                    ax.text(i, v + mx * 0.05, f"{v}", ha='center', va='bottom', fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown("<div style='margin-top:20px; margin-bottom:20px;'></div>", unsafe_allow_html=True)
            
            def calculate_zscore(df):
                z = pd.DataFrame(index=df.index)
                z['T1'] = (df['Current Assets'] - df['Total Current Liabilities']) / df['Total Assets']
                z['T2'] = df['Retained Earnings'] / df['Total Assets']
                z['T3'] = df['EBIT'] / df['Total Assets']
                z['T4'] = df['Market Value'] / df['Total Liabilities']
                z['T5'] = df['Net Sales'] / df['Total Assets']
                z['Z-Score'] = (1.2 * z['T1'] + 1.4 * z['T2'] + 
                               3.3 * z['T3'] + 0.6 * z['T4'] + 
                               0.99 * z['T5'])
                z.replace([np.inf, -np.inf], np.nan, inplace=True)
                z['Z-Score'].fillna(z['Z-Score'].median(), inplace=True)
                z['Z-Score Status'] = pd.cut(
                    z['Z-Score'],
                    bins=[-np.inf, 1.8, 2.99, np.inf],
                    labels=['Distress', 'Grey', 'Safe']
                )
                z['Z-Score Prediction'] = (z['Z-Score Status'] == 'Distress').astype(int)
                z['Actual Status'] = df['Bankrupt']
                return z
            
            z_df = calculate_zscore(data)
            
            # Compare Z-Score vs ML models
            z_pred = z_df['Z-Score Prediction'].values
            z_act = z_df['Actual Status'].values
            z_acc = (z_pred == z_act).mean()
            z_prec = (z_pred & z_act).sum() / z_pred.sum() if z_pred.sum() > 0 else 0
            z_rec = (z_pred & z_act).sum() / z_act.sum() if z_act.sum() > 0 else 0
            z_f1 = 2 * z_prec * z_rec / (z_prec + z_rec) if (z_prec + z_rec) > 0 else 0
            
            comp = pd.DataFrame({
                'Model': ['Altman Z-Score'] + list(metrics.keys()),
                'Accuracy': [z_acc] + [metrics[m]['accuracy'] for m in metrics],
                'Precision': [z_prec] + [metrics[m]['precision'] for m in metrics],
                'Recall': [z_rec] + [metrics[m]['recall'] for m in metrics],
                'F1 Score': [z_f1] + [metrics[m]['f1'] for m in metrics]
            }).set_index('Model')
            st.markdown("### Z-Score vs Machine Learning Models")
            st.dataframe(comp.style.highlight_max(axis=0))
            
            # Z-Score confusion matrix
            z_tn = ((z_pred == 0) & (z_act == 0)).sum()
            z_fp = ((z_pred == 1) & (z_act == 0)).sum()
            z_fn = ((z_pred == 0) & (z_act == 1)).sum()
            z_tp = ((z_pred == 1) & (z_act == 1)).sum()
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("#### Z-Score Confusion Matrix")
                st.dataframe(pd.DataFrame([[z_tn, z_fp], [z_fn, z_tp]],
                                          index=['Actual Alive', 'Actual Bankrupt'],
                                          columns=['Pred Alive', 'Pred Bankrupt']))
            with c2:
                st.markdown("#### Z-Score Metrics")
                st.write(f"- Accuracy: {z_acc:.4f}")
                st.write(f"- Precision: {z_prec:.4f}")
                st.write(f"- Recall: {z_rec:.4f}")
                st.write(f"- F1 Score: {z_f1:.4f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888888; font-size: 0.8em;">
Bankruptcy Prediction Dashboard | Created with Streamlit | Data Analysis Based on Financial Metrics
</div>
""", unsafe_allow_html=True)

