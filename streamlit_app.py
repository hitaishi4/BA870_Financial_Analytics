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
                
                # Handle status_label → Bankrupt mapping
                if "status_label" in df.columns:
                    status_values = df['status_label'].unique()
                    st.sidebar.info(f"Status values found: {', '.join(status_values)}")
                    if 'failed' in status_values:
                        df['Bankrupt'] = df['status_label'].map({'failed': 1, 'alive': 0})
                    elif 'Bankrupt' in status_values:
                        df['Bankrupt'] = df['status_label'].map({'Bankrupt': 1, 'Alive': 0})
                    else:
                        df['Bankrupt'] = df['status_label'].apply(
                            lambda x: 1 if x.lower() in ['failed','bankrupt','distress','default'] else 0
                        )
                    st.sidebar.success("✅ Converted status_label to Bankrupt column")
                
                # Rename if there's a “Bankruptcy” column
                if "Bankruptcy" in df.columns and "Bankrupt" not in df.columns:
                    df['Bankrupt'] = df['Bankruptcy']
                    st.sidebar.success("✅ Renamed Bankruptcy column to Bankrupt")
                
                # Rename X1–X18
                if "X1" in df.columns:
                    df = df.rename(columns=rename_map)
                    st.sidebar.success("✅ Renamed X1–X18 columns to descriptive names")
                
                # ---- NEW: drop any non-numeric columns except status_label ----
                for col in df.select_dtypes(include=['object']).columns:
                    if col != "status_label":
                        df.drop(columns=col, inplace=True)
                st.sidebar.success("✅ Dropped non-numeric identifier columns")
                # ---------------------------------------------------------------

                return df
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            continue

    st.sidebar.error("❌ Failed to load data from any location.")
    st.error("Could not load the data file. Please check that the file exists in the specified locations.")
    return pd.DataFrame()

# Load data and show sidebar info
try:
    data = load_data()
    if not data.empty:
        with st.sidebar.expander("📊 Data Information"):
            st.write(f"Rows: {data.shape[0]}")
            st.write(f"Columns: {data.shape[1]}")
            st.write("Column Names:")
            st.write(", ".join(data.columns.tolist()))
            
            required_cols = [
                'Current Assets','Total Current Liabilities','Retained Earnings',
                'Total Assets','EBIT','Market Value','Total Liabilities','Net Sales'
            ]
            missing_cols = [c for c in required_cols if c not in data.columns]
            if missing_cols:
                st.error(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ All required columns for Z-Score calculation are present")
except Exception as e:
    st.error(f"Error during data initialization: {e}")
    data = pd.DataFrame()

st.session_state['data_loaded'] = not data.empty

# Define feature names & precomputed model metrics
feature_names = [
    "Current Assets","Cost of Goods Sold","D&A","EBITDA","Inventory","Net Income",
    "Total Receivables","Market Value","Net Sales","Total Assets",
    "Total Long-term Debt","EBIT","Gross Profit","Total Current Liabilities",
    "Retained Earnings","Total Revenue","Total Liabilities","Total Operating Expenses"
]

metrics = {
    'Decision Tree': {
        'accuracy':0.8925,'precision':0.0589,'recall':0.2404,'f1':0.0947,'auc':0.574,
        'confusion_matrix':[[10893,1102],[218,69]]
    },
    'Gradient Boosting': {
        'accuracy':0.9761,'precision':0.3846,'recall':0.0348,'f1':0.0639,'auc':0.827,
        'confusion_matrix':[[11979,16],[277,10]]
    },
    'Random Forest': {
        'accuracy':0.9759,'precision':0.3200,'recall':0.0279,'f1':0.0513,'auc':0.835,
        'confusion_matrix':[[11978,17],[279,8]]
    },
    'Logistic Regression': {
        'accuracy':0.9752,'precision':0.3125,'recall':0.0523,'f1':0.0896,'auc':0.827,
        'confusion_matrix':[[11962,33],[272,15]]
    },
    'SVM': {
        'accuracy':0.9765,'precision':0.3333,'recall':0.0070,'f1':0.0137,'auc':0.590,
        'confusion_matrix':[[11991,4],[285,2]]
    },
    'KNN': {
        'accuracy':0.9589,'precision':0.1414,'recall':0.1498,'f1':0.1455,'auc':0.695,
        'confusion_matrix':[[11734,261],[244,43]]
    }
}

feature_importances = {
    'Decision Tree': {
        'Retained Earnings':0.072059,'Market Value':0.072055,'Inventory':0.070231,
        'D&A':0.068246,'Gross Profit':0.067548,'Total Receivables':0.065696,
        'Current Assets':0.065387,'Total Long-term Debt':0.064578,'Total Assets':0.056883,
        'Total Current Liabilities':0.055932,'Net Income':0.055526,'Total Liabilities':0.052951,
        'Cost of Goods Sold':0.051296,'Total Operating Expenses':0.047349,'EBITDA':0.041733,
        'EBIT':0.041661,'Total Revenue':0.027468,'Net Sales':0.023400
    },
    'Gradient Boosting': {
        'Total Long-term Debt':0.115407,'Net Income':0.113170,'Retained Earnings':0.088011,
        'Market Value':0.083996,'Inventory':0.075858,'Total Operating Expenses':0.071508,
        'Current Assets':0.068556,'Total Receivables':0.066965,'Gross Profit':0.056605,
        'D&A':0.045299,'Total Liabilities':0.040103,'EBITDA':0.031667,'EBIT':0.030457,
        'Net Sales':0.028807,'Cost of Goods Sold':0.028534,'Total Current Liabilities':0.022211,
        'Total Assets':0.017786,'Total Revenue':0.015061
    },
    'Random Forest': {
        'Retained Earnings':0.065674,'Market Value':0.062897,'D&A':0.061341,'Current Assets':0.059910,
        'Total Receivables':0.059713,'Gross Profit':0.058533,'Total Liabilities':0.057575,
        'Total Assets':0.057426,'Total Current Liabilities':0.055479,'Inventory':0.054929,
        'Total Long-term Debt':0.054677,'Net Income':0.053633,'Cost of Goods Sold':0.053133,
        'EBITDA':0.051601,'EBIT':0.050618,'Total Operating Expenses':0.049919,
        'Total Revenue':0.046852,'Net Sales':0.046092
    },
    'Logistic Regression': {
        'Market Value':1.102307,'Current Assets':0.976875,'Total Current Liabilities':0.500875,
        'EBIT':0.418057,'Total Long-term Debt':0.366918,'Total Liabilities':0.335098,
        'EBITDA':0.309482,'Inventory':0.285948,'Total Assets':0.231877,
        'Gross Profit':0.153693,'Cost of Goods Sold':0.065107,'Total Operating Expenses':0.056967,
        'Retained Earnings':0.054134,'Total Receivables':0.040750,'Net Income':0.019487,
        'D&A':0.006214,'Net Sales':0.001644,'Total Revenue':0.001644
    },
    'KNN': {
        'Inventory':0.048982,'D&A':0.048754,'Total Long-term Debt':0.042688,'Gross Profit':0.039603,
        'Retained Earnings':0.030695,'Total Liabilities':0.023482,'Cost of Goods Sold':0.005708,
        'EBIT':0.004975,'Total Operating Expenses':0.001930,'Total Revenue':0.001262,
        'Net Sales':0.001262,'Total Current Liabilities':0.000244,'Current Assets':-0.000090,
        'Total Receivables':-0.000627,'Total Assets':-0.001449,'EBITDA':-0.001767,
        'Market Value':-0.002597,'Net Income':-0.004633
    },
    'SVM': {
        'Current Assets':0.000147,'Total Receivables':0.000090,'Gross Profit':0.000008,'Total Revenue':0.000000,
        'Cost of Goods Sold':0.000000,'Net Sales':0.000000,'Total Assets':0.000000,'EBITDA':0.000000,
        'D&A':0.000000,'Total Operating Expenses':0.000000,'Market Value':-0.000008,'Inventory':-0.000008,
        'Total Current Liabilities':-0.000008,'Net Income':-0.000016,'EBIT':-0.000016,
        'Total Long-term Debt':-0.000090,'Total Liabilities':-0.000163,'Retained Earnings':-0.000220
    }
}

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
# Sidebar navigation
st.sidebar.title("Navigation")
pages = [
    "Overview", "Model Comparison", "ROC Curves",
    "Feature Importance", "Confusion Matrices", "Z-Score Analysis"
]
selected_page = st.sidebar.radio("Go to", pages)

# Render the selected page
if selected_page == "Overview":
    st.markdown('<p class="sub-header">Overview</p>', unsafe_allow_html=True)
    st.markdown("""
    ### Project Summary
    This dashboard presents the results of a bankruptcy prediction analysis...
    """)
    st.markdown('<p class="section-header">Performance Summary</p>', unsafe_allow_html=True)
    # Create and display metrics_df
    metrics_df = pd.DataFrame({
        'Accuracy': [metrics[m]['accuracy'] for m in metrics],
        'Precision': [metrics[m]['precision'] for m in metrics],
        'Recall': [metrics[m]['recall'] for m in metrics],
        'F1 Score': [metrics[m]['f1'] for m in metrics],
        'AUC': [metrics[m]['auc'] for m in metrics]
    }, index=metrics.keys())
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best AUC",
                  f"{metrics_df['AUC'].max():.3f}",
                  metrics_df['AUC'].idxmax())
    with col2:
        st.metric("Best F1 Score",
                  f"{metrics_df['F1 Score'].max():.3f}",
                  metrics_df['F1 Score'].idxmax())
    with col3:
        st.metric("Best Recall",
                  f"{metrics_df['Recall'].max():.3f}",
                  metrics_df['Recall'].idxmax())
    st.markdown("### Quick insights")
    st.markdown(f"""
    - Best overall (AUC): **{metrics_df['AUC'].idxmax()}** ({metrics_df['AUC'].max():.3f})
    - Best recall: **{metrics_df['Recall'].idxmax()}** ({metrics_df['Recall'].max():.3f})
    - Best precision: **{metrics_df['Precision'].idxmax()}** ({metrics_df['Precision'].max():.3f})
    """)
    # AUC bar chart
    auc_series = metrics_df['AUC'].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(auc_series.index, auc_series.values, color='#395c40')
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom')
    ax.set_ylabel('AUC Score')
    ax.set_title('Model AUC Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    # Dataset preview & stats
    if st.session_state['data_loaded']:
        st.markdown("### Dataset Preview")
        preview_data = data.copy()
        display_cols = []
        if 'status_label' in preview_data: display_cols.append('status_label')
        if 'Bankrupt' in preview_data: display_cols.append('Bankrupt')
        if 'year' in preview_data: display_cols.append('year')
        for col in ['Current Assets','Total Assets','Net Income','EBIT','Market Value']:
            if col in preview_data: display_cols.append(col)
        if not display_cols and not preview_data.empty:
            display_cols = preview_data.columns[:5].tolist()
        st.dataframe(preview_data[display_cols].head())
        st.markdown("### Dataset Statistics")
        st.write(f"Number of records: {len(data)}")
        st.write(f"Number of features: {len(data.columns)}")
        if 'Bankrupt' in data:
            bankruptcy_counts = data['Bankrupt'].value_counts().reset_index()
            bankruptcy_counts.columns = ['Status','Count']
            bankruptcy_counts['Status'] = bankruptcy_counts['Status'].map({1:'Bankrupt',0:'Healthy'})
            fig, ax = plt.subplots(figsize=(8,6))
            ax.pie(bankruptcy_counts['Count'],
                   labels=bankruptcy_counts['Status'],
                   autopct='%1.1f%%', startangle=90,
                   colors=['#a63603','#395c40'])
            ax.axis('equal')
            plt.title('Distribution of Bankruptcy Status')
            st.pyplot(fig)

elif selected_page == "Model Comparison":
    st.markdown('<p class="sub-header">Model Performance Comparison</p>', unsafe_allow_html=True)
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
    metric_options = ["Accuracy","Precision","Recall","F1 Score","AUC"]
    selected_metrics = st.multiselect("Select metrics to compare", metric_options, default=["Recall","F1 Score","AUC"])
    if selected_metrics:
        fig, axes = plt.subplots(1, len(selected_metrics), figsize=(15,5))
        if len(selected_metrics)==1: axes=[axes]
        for i, metric in enumerate(selected_metrics):
            sorted_df = metrics_df.sort_values(metric, ascending=False)
            bars = axes[i].bar(sorted_df.index, sorted_df[metric], color='#395c40')
            for bar in bars:
                axes[i].text(bar.get_x()+bar.get_width()/2,
                             bar.get_height()+0.01,
                             f'{bar.get_height():.3f}',
                             ha='center',va='bottom')
            axes[i].set_title(metric)
            axes[i].set_ylim(0, sorted_df[metric].max()*1.2)
            axes[i].tick_params(axis='x',rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    st.markdown("""
    ### Understanding the Metrics
    - **Accuracy**: …  
    - **Precision**: …  
    - **Recall**: …  
    - **F1 Score**: …  
    - **AUC**: …  
    """)
    st.markdown("### Class Imbalance")
    st.info("""
    **Note on Class Imbalance**: The dataset has a significant class imbalance…  
    """)

elif selected_page == "ROC Curves":
    st.markdown('<p class="sub-header">ROC Curve Analysis</p>', unsafe_allow_html=True)
    st.markdown("""
    ### What are ROC Curves?
    ROC curves plot…  
    """)
    model_options = list(metrics.keys())
    selected_models = st.multiselect("Select models to compare", model_options, default=model_options[:3])
    if selected_models:
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot([0,1],[0,1], linestyle='--', color='gray', alpha=0.8, label='Random')
        colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
        for i, model in enumerate(selected_models):
            fpr = roc_curves[model]['fpr']
            tpr = roc_curves[model]['tpr']
            auc = roc_curves[model]['auc']
            ax.plot(fpr, tpr, lw=2, color=colors[i%len(colors)], label=f'{model} (AUC={auc:.3f})')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend(loc='lower right')
        ax.set_xlim([0.0,1.0])
        ax.set_ylim([0.0,1.05])
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("### Individual Model ROC Curve")
    single_model = st.selectbox("Select a model", options=model_options)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot([0,1],[0,1], linestyle='--', color='gray', alpha=0.8, label='Random')
    fpr = roc_curves[single_model]['fpr']
    tpr = roc_curves[single_model]['tpr']
    auc = roc_curves[single_model]['auc']
    ax.plot(fpr, tpr, lw=2, color='#395c40', label=f'ROC (AUC={auc:.3f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{single_model} ROC Curve')
    ax.legend(loc='lower right')
    ax.set_xlim([0.0,1.0])
    ax.set_ylim([0.0,1.05])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
    ### Interpreting ROC Curves
    - **AUC**: …  
    - **Thresholds**: …  
    - **Optimal Threshold**: …  
    """)
elif selected_page == "Feature Importance":
    st.markdown('<p class="sub-header">Feature Importance Analysis</p>', unsafe_allow_html=True)
    model_options = ["Decision Tree","Gradient Boosting","Random Forest","Logistic Regression","KNN","SVM"]
    selected_model = st.selectbox("Select model", model_options)
    importances = pd.Series(feature_importances[selected_model]).sort_values(ascending=False)
    n_features = st.slider("Number of top features to display", 5, len(feature_names), 10)
    st.markdown(f"### Top {n_features} Features for {selected_model}")
    fig, ax = plt.subplots(figsize=(10,8))
    top_feats = importances.head(n_features)
    max_val = importances.max()
    bars = ax.barh(top_feats.index[::-1], top_feats.values[::-1], color='#395c40')
    for bar in bars:
        ax.text(bar.get_width()+max_val*0.05,
                bar.get_y()+bar.get_height()/2,
                f'{bar.get_width():.3f}',
                va='center', ha='left')
    ax.set_xlim(0, max_val*1.25)
    ax.set_xlabel('Importance')
    ax.set_title(f'{selected_model} Feature Importance')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Top 5 Features Across Models")
    comparison_models = ["Decision Tree","Gradient Boosting","Random Forest","Logistic Regression"]
    comp_df = pd.DataFrame(index=range(1,6), columns=comparison_models)
    for m in comparison_models:
        idxs = pd.Series(feature_importances[m]).sort_values(ascending=False).index
        for i in range(5):
            comp_df.loc[i+1,m] = idxs[i]
    st.dataframe(comp_df)
    st.markdown("""
    ### Interpreting Feature Importance
    - **Decision Tree**: …  
    - **Gradient Boosting & Random Forest**: …  
    - **Logistic Regression**: …  
    - **KNN & SVM**: …  
    """)

elif selected_page == "Confusion Matrices":
    st.markdown('<p class="sub-header">Confusion Matrix Analysis</p>', unsafe_allow_html=True)
    model_options = list(metrics.keys())
    selected_model = st.selectbox("Select model", model_options)
    cm = metrics[selected_model]['confusion_matrix']
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn+fp+fn+tp
    accuracy = (tn+tp)/total
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0

    st.markdown(f"### {selected_model} Confusion Matrix")
    col1, col2 = st.columns([2,1])
    with col1:
        cm_df = pd.DataFrame(cm,
                             index=['Actual Alive','Actual Bankrupt'],
                             columns=['Predicted Alive','Predicted Bankrupt'])
        st.dataframe(cm_df)
        # visual boxes
        cm_pct = np.array([[100*tn/(tn+fp) if tn+fp else 0,
                            100*fp/(tn+fp) if tn+fp else 0],
                           [100*fn/(fn+tp) if fn+tp else 0,
                            100*tp/(fn+tp) if fn+tp else 0]])
        html = f"""
        <style>
        .cm-box {{ padding:20px; text-align:center; font-weight:bold; color:white; }}
        .box-container {{ display:grid; grid-template:1fr 1fr / 1fr 1fr; gap:10px; }}
        .tn {{ background-color:rgba(57,92,64,0.8); }}
        .fp, .fn {{ background-color:rgba(166,54,3,0.8); }}
        .tp {{ background-color:rgba(57,92,64,0.8); }}
        </style>
        <div class="box-container">
          <div class="cm-box tn">TN<br>{tn} ({cm_pct[0,0]:.1f}%)</div>
          <div class="cm-box fp">FP<br>{fp} ({cm_pct[0,1]:.1f}%)</div>
          <div class="cm-box fn">FN<br>{fn} ({cm_pct[1,0]:.1f}%)</div>
          <div class="cm-box tp">TP<br>{tp} ({cm_pct[1,1]:.1f}%)</div>
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
    comp = pd.DataFrame(
        index=model_options,
        columns=["True Positives","False Negatives","Detection Rate (%)","False Alarm Rate (%)"]
    )
    for m in model_options:
        tn,fp = metrics[m]['confusion_matrix'][0]
        fn,tp = metrics[m]['confusion_matrix'][1]
        comp.loc[m,"True Positives"] = tp
        comp.loc[m,"False Negatives"] = fn
        comp.loc[m,"Detection Rate (%)"] = 100*tp/(tp+fn) if tp+fn else 0
        comp.loc[m,"False Alarm Rate (%)"] = 100*fp/(tn+fp) if tn+fp else 0
    comp = comp.sort_values("Detection Rate (%)", ascending=False)
    st.dataframe(comp)
    st.info("""
    A higher detection rate often comes with more false alarms…  
    """)

elif selected_page == "Z-Score Analysis":
    st.markdown('<p class="sub-header">Altman Z-Score Analysis</p>', unsafe_allow_html=True)
    st.markdown("""
    ### What is the Altman Z-Score?
    Z = 1.2*T1 + 1.4*T2 + 3.3*T3 + 0.6*T4 + 0.99*T5  
    T1 = (Current Assets – Current Liabilities)/Total Assets  
    T2 = Retained Earnings/Total Assets  
    T3 = EBIT/Total Assets  
    T4 = Market Value/Total Liabilities  
    T5 = Net Sales/Total Assets  
    Interpretation: Z>2.99=Safe; 1.8<Z<2.99=Grey; Z<1.8=Distress
    """)
    if st.session_state['data_loaded'] and not data.empty:
        missing = [c for c in [
            'Current Assets','Total Current Liabilities','Retained Earnings',
            'Total Assets','EBIT','Market Value','Total Liabilities','Net Sales'
        ] if c not in data.columns]
        if missing:
            st.error(f"Cannot calculate Z-Score: Missing {', '.join(missing)}")
            st.warning("Ensure all financial metrics are present.")
            with st.expander("Current Columns & Mapping"):
                st.write(", ".join(data.columns))
                mapping_df = pd.DataFrame(list(rename_map.items()), columns=["Original","Expected"])
                st.dataframe(mapping_df)
        else:
            # compute bankruptcy rate bar chart
            bankrupt_count = data['Bankrupt'].sum()
            alive_count = len(data)-bankrupt_count
            fig, ax = plt.subplots(figsize=(8,5))
            bars = ax.bar(['Alive','Bankrupt'], [alive_count,bankrupt_count], color=['#395c40','#a63603'])
            ax.set_ylim(0, max(alive_count,bankrupt_count)*1.15)
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.05,
                        f"{int(bar.get_height()):,}", ha='center', va='bottom', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Companies')
            plt.tight_layout()
            st.pyplot(fig)
            st.info(f"**Bankruptcy Rate**: {100*bankrupt_count/len(data):.2f}% ({bankrupt_count:,}/{len(data):,})")
            if bankrupt_count==0:
                st.warning("⚠️ No bankruptcies found—check your 'Bankrupt' column.")

            # Z-Score calculation function
            def calculate_custom_zscore():
                z_df = pd.DataFrame(index=data.index)
                z_df['T1'] = (data['Current Assets'] - data['Total Current Liabilities'])/data['Total Assets']
                z_df['T2'] = data['Retained Earnings']/data['Total Assets']
                z_df['T3'] = data['EBIT']/data['Total Assets']
                z_df['T4'] = data['Market Value']/data['Total Liabilities']
                z_df['T5'] = data['Net Sales']/data['Total Assets']
                z_df['Z-Score'] = (1.2*z_df['T1'] + 1.4*z_df['T2'] + 3.3*z_df['T3'] +
                                   0.6*z_df['T4'] + 0.99*z_df['T5'])
                z_df.replace([np.inf,-np.inf], np.nan, inplace=True)
                z_df['Z-Score'].fillna(z_df['Z-Score'].median(), inplace=True)
                thresholds = (1.8,2.99)
                z_df['Z-Score Status'] = pd.cut(
                    z_df['Z-Score'], bins=[-np.inf,*thresholds,np.inf],
                    labels=['Distress','Grey','Safe']
                )
                z_df['Z-Score Prediction'] = (z_df['Z-Score Status']=='Distress').astype(int)
                z_df['Actual Status'] = data['Bankrupt']
                return z_df

            zscore_df = calculate_custom_zscore()
            if not zscore_df.empty:
                with st.expander("Z-Score Statistics"):
                    st.write(f"Mean: {zscore_df['Z-Score'].mean():.4f}")
                    st.write(f"Median: {zscore_df['Z-Score'].median():.4f}")
                    st.write(f"Min: {zscore_df['Z-Score'].min():.4f}")
                    st.write(f"Max: {zscore_df['Z-Score'].max():.4f}")
                    counts = zscore_df['Z-Score Status'].value_counts()
                    for zone in counts.index:
                        st.write(f"- {zone}: {counts[zone]:,}")

                # compare with ML models
                z_pred = zscore_df['Z-Score Prediction']
                z_act = zscore_df['Actual Status']
                z_acc = (z_pred==z_act).mean()
                z_prec = (z_pred & z_act).sum()/z_pred.sum() if z_pred.sum() else 0
                z_rec = (z_pred & z_act).sum()/z_act.sum() if z_act.sum() else 0
                z_f1 = 2*z_prec*z_rec/(z_prec+z_rec) if (z_prec+z_rec) else 0
                comparison_data = {
                    'Model': ['Altman Z-Score']+list(metrics.keys()),
                    'Accuracy': [z_acc]+[metrics[m]['accuracy'] for m in metrics],
                    'Precision': [z_prec]+[metrics[m]['precision'] for m in metrics],
                    'Recall': [z_rec]+[metrics[m]['recall'] for m in metrics],
                    'F1 Score': [z_f1]+[metrics[m]['f1'] for m in metrics]
                }
                comp_df = pd.DataFrame(comparison_data).set_index('Model')
                st.markdown("### Z-Score vs. ML Models")
                st.dataframe(comp_df.style.highlight_max(axis=0))

                # Z-Score confusion matrix
                z_tn = ((z_pred==0)&(z_act==0)).sum()
                z_fp = ((z_pred==1)&(z_act==0)).sum()
                z_fn = ((z_pred==0)&(z_act==1)).sum()
                z_tp = ((z_pred==1)&(z_act==1)).sum()
                col1, col2 = st.columns([2,1])
                with col1:
                    z_cm_df = pd.DataFrame([[z_tn,z_fp],[z_fn,z_tp]],
                                            index=['Actual Alive','Actual Bankrupt'],
                                            columns=['Pred Alive','Pred Bankrupt'])
                    st.dataframe(z_cm_df)
                with col2:
                    st.markdown("### Z-Score Metrics")
                    st.markdown(f"""
                    - Accuracy: {z_acc:.4f}
                    - Precision: {z_prec:.4f}
                    - Recall: {z_rec:.4f}
                    - F1 Score: {z_f1:.4f}
                    """)
            else:
                st.warning("Could not calculate Z-Score. Check your data.")
    else:
        st.error("Please upload your data file named 'american_bankruptcy.csv' in 'data/'.")
        st.info("""
        Dataset must include:
        - Current Assets
        - Total Current Liabilities
        - Retained Earnings
        - Total Assets
        - EBIT
        - Market Value
        - Total Liabilities
        - Net Sales
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:0.8em;">
Bankruptcy Prediction Dashboard | Created with Streamlit | Data Analysis Based on Financial Metrics
</div>
""", unsafe_allow_html=True)
