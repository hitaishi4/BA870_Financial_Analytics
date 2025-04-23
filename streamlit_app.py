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

# Improved data loading function with better error handling and debug information
@st.cache_data
def load_data():
    """Load data with multiple fallback paths and detailed error reporting"""
    # List of possible file paths to try
    possible_paths = [
        'data/american_bankruptcy.csv',  # Default path in GitHub repo
        'american_bankruptcy.csv',       # Root directory
        '../data/american_bankruptcy.csv', # One level up
        './american_bankruptcy.csv',     # Explicit current directory
    ]
    
    # Try each path
    for path in possible_paths:
        try:
            st.sidebar.info(f"Trying to load from: {path}")
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Data loaded successfully from {path}")
                return df
        except Exception as e:
            continue
    
    # If we reach here, all paths failed
    st.sidebar.error("❌ Failed to load data from any location.")
    st.error("Could not load the data file. Please check that the file exists in the specified locations.")
    return pd.DataFrame()  # Return empty DataFrame if all paths fail

# Load the data and display debug information in the sidebar
try:
    data = load_data()
    
    # Display data info in sidebar only if data was successfully loaded
    if not data.empty:
        with st.sidebar.expander("📊 Data Information"):
            st.write(f"Rows: {data.shape[0]}")
            st.write(f"Columns: {data.shape[1]}")
            st.write("Column Names:")
            st.write(", ".join(data.columns.tolist()))
            
            # Check for required columns
            required_cols = ['Current Assets', 'Total Current Liabilities', 'Retained Earnings', 
                            'Total Assets', 'EBIT', 'Market Value', 'Total Liabilities', 'Net Sales']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                st.error(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ All required columns for Z-Score calculation are present")
except Exception as e:
    st.error(f"Error during data initialization: {str(e)}")
    data = pd.DataFrame()

# Set session state to track data loading
st.session_state['data_loaded'] = not data.empty

# Define feature names and model results based on your analysis
feature_names = [
    "Current Assets", "Cost of Goods Sold", "D&A", "EBITDA",
    "Inventory", "Net Income", "Total Receivables", "Market Value",
    "Net Sales", "Total Assets", "Total Long-term Debt", "EBIT",
    "Gross Profit", "Total Current Liabilities", "Retained Earnings",
    "Total Revenue", "Total Liabilities", "Total Operating Expenses"
]

# Define metrics from your analysis with revised AUC values
metrics = {
    'Decision Tree': {
        'accuracy': 0.8925,
        'precision': 0.0589,
        'recall': 0.2404,
        'f1': 0.0947,
        'auc': 0.574,  # Updated AUC value
        'confusion_matrix': [[10893, 1102], [218, 69]]
    },
    'Gradient Boosting': {
        'accuracy': 0.9761,
        'precision': 0.3846,
        'recall': 0.0348,
        'f1': 0.0639,
        'auc': 0.827,  # Updated AUC value
        'confusion_matrix': [[11979, 16], [277, 10]]
    },
    'Random Forest': {
        'accuracy': 0.9759,
        'precision': 0.3200,
        'recall': 0.0279,
        'f1': 0.0513,
        'auc': 0.835,  # Updated AUC value
        'confusion_matrix': [[11978, 17], [279, 8]]
    },
    'Logistic Regression': {
        'accuracy': 0.9752,
        'precision': 0.3125,
        'recall': 0.0523,
        'f1': 0.0896,
        'auc': 0.827,  # Updated AUC value
        'confusion_matrix': [[11962, 33], [272, 15]]
    },
    'SVM': {
        'accuracy': 0.9765,
        'precision': 0.3333,
        'recall': 0.0070,
        'f1': 0.0137,
        'auc': 0.590,  # Updated AUC value
        'confusion_matrix': [[11991, 4], [285, 2]]
    },
    'KNN': {
        'accuracy': 0.9589,
        'precision': 0.1414,
        'recall': 0.1498,
        'f1': 0.1455,
        'auc': 0.695,  # Updated AUC value
        'confusion_matrix': [[11734, 261], [244, 43]]
    }
}

# Feature importance data from your analysis
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

# Function to calculate Z-Score
def calculate_zscore(df):
    """Calculate Altman Z-Score for financial data"""
    # Create a Z-Score dataframe
    zscore_df = pd.DataFrame(index=df.index)
    
    try:
        # T1 = Working Capital / Total Assets
        zscore_df['T1'] = (df['Current Assets'] - df['Total Current Liabilities']) / df['Total Assets']
        
        # T2 = Retained Earnings / Total Assets
        zscore_df['T2'] = df['Retained Earnings'] / df['Total Assets']
        
        # T3 = EBIT / Total Assets
        zscore_df['T3'] = df['EBIT'] / df['Total Assets']
        
        # T4 = Market Value / Total Liabilities
        zscore_df['T4'] = df['Market Value'] / df['Total Liabilities']
        
        # T5 = Sales / Total Assets
        zscore_df['T5'] = df['Net Sales'] / df['Total Assets']
        
        # Calculate Z-Score
        zscore_df['Z-Score'] = (1.2 * zscore_df['T1'] + 
                              1.4 * zscore_df['T2'] + 
                              3.3 * zscore_df['T3'] + 
                              0.6 * zscore_df['T4'] + 
                              0.99 * zscore_df['T5'])
        
        # Classify based on Z-Score
        zscore_df['Z-Score Status'] = pd.cut(
            zscore_df['Z-Score'], 
            bins=[-float('inf'), 1.8, 2.99, float('inf')],
            labels=['Distress', 'Grey', 'Safe']
        )
        
        # Convert Z-Score classification to binary (Distress = 1, others = 0)
        zscore_df['Z-Score Prediction'] = (zscore_df['Z-Score Status'] == 'Distress').astype(int)
        
        return zscore_df
    except Exception as e:
        st.error(f"Error calculating Z-Score: {e}")
        return pd.DataFrame()

# Render the selected page
if selected_page == "Overview":
    st.markdown('<p class="sub-header">Overview</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Summary
    
    This dashboard presents the results of a bankruptcy prediction analysis using financial data
    from American companies. The dataset includes 18 financial features and spans multiple years.
    
    ### Methodology
    
    - **Training Data**: Financial data from 1999-2011
    - **Testing Data**: Financial data from 2015-2018
    - **Features**: 18 financial indicators including Current Assets, Net Income, EBITDA, etc.
    - **Target Variable**: Binary classification (Bankrupt vs Alive)
    
    ### Models Analyzed
    
    - Decision Tree
    - Gradient Boosting
    - Random Forest
    - Logistic Regression
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    
    ### Key Metrics
    
    - Accuracy, Precision, Recall, F1 Score
    - ROC Curves and AUC
    - Confusion Matrices
    - Feature Importance
    """)
    
    # Display summary of results
    st.markdown('<p class="section-header">Performance Summary</p>', unsafe_allow_html=True)
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Accuracy': [metrics[model]['accuracy'] for model in metrics],
        'Precision': [metrics[model]['precision'] for model in metrics],
        'Recall': [metrics[model]['recall'] for model in metrics],
        'F1 Score': [metrics[model]['f1'] for model in metrics],
        'AUC': [metrics[model]['auc'] for model in metrics]
    }, index=metrics.keys())
    
    # Display in 3 columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best AUC", 
                  f"{metrics_df['AUC'].max():.3f}", 
                  f"{metrics_df['AUC'].idxmax()}")
    
    with col2:
        st.metric("Best F1 Score", 
                  f"{metrics_df['F1 Score'].max():.3f}", 
                  f"{metrics_df['F1 Score'].idxmax()}")
    
    with col3:
        st.metric("Best Recall", 
                  f"{metrics_df['Recall'].max():.3f}", 
                  f"{metrics_df['Recall'].idxmax()}")
    
    st.markdown("### Quick insights")
    st.markdown(f"""
    - The model with the best overall performance is **{metrics_df['AUC'].idxmax()}** with an AUC of {metrics_df['AUC'].max():.3f}
    - For identifying bankruptcies (recall), **{metrics_df['Recall'].idxmax()}** performs best with a recall of {metrics_df['Recall'].max():.3f}
    - The highest precision is achieved by **{metrics_df['Precision'].idxmax()}** at {metrics_df['Precision'].max():.3f}
    """)

    # Plot AUC comparison
    st.markdown("### Model AUC Comparison")
    auc_series = metrics_df['AUC'].sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(auc_series.index, auc_series.values, color='#395c40')
    
    # Add values to bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom'
        )
    
    ax.set_ylabel('AUC Score')
    ax.set_title('Model AUC Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Display dataset info if data is loaded
    if st.session_state.get('data_loaded', False):
        st.markdown("### Dataset Preview")
        st.dataframe(data.head())
        
        st.markdown("### Dataset Statistics")
        st.write(f"Number of records: {len(data)}")
        st.write(f"Number of features: {len(data.columns)}")

elif selected_page == "Model Comparison":
    st.markdown('<p class="sub-header">Model Performance Comparison</p>', unsafe_allow_html=True)
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Accuracy': [metrics[model]['accuracy'] for model in metrics],
        'Precision': [metrics[model]['precision'] for model in metrics],
        'Recall': [metrics[model]['recall'] for model in metrics],
        'F1 Score': [metrics[model]['f1'] for model in metrics],
        'AUC': [metrics[model]['auc'] for model in metrics]
    }, index=metrics.keys())
    
    # Display metrics table
    st.markdown("### Performance Metrics")
    st.dataframe(metrics_df.style.highlight_max(axis=0))
    
    # Select metrics to visualize
    st.markdown("### Metric Comparison")
    metric_options = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    selected_metrics = st.multiselect(
        "Select metrics to compare", 
        options=metric_options,
        default=["Recall", "F1 Score", "AUC"]
    )
    
    if selected_metrics:
        # Create subplot for each selected metric
        fig, axes = plt.subplots(1, len(selected_metrics), figsize=(15, 5))
        
        # Handle case when only one metric is selected
        if len(selected_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(selected_metrics):
            # Sort by metric value
            sorted_df = metrics_df.sort_values(metric, ascending=False)
            
            # Create bar chart
            bars = axes[i].bar(sorted_df.index, sorted_df[metric], color='#395c40')
            
            # Add values to bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.01,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom'
                )
            
            axes[i].set_title(metric)
            axes[i].set_ylim(0, sorted_df[metric].max() * 1.2)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Add detailed explanation
    st.markdown("""
    ### Understanding the Metrics
    
    - **Accuracy**: The ratio of correctly predicted instances to the total instances. High accuracy can be misleading with imbalanced classes.
    
    - **Precision**: The ratio of correctly predicted positive instances to the total predicted positive instances. High precision means low false positive rate.
    
    - **Recall**: The ratio of correctly predicted positive instances to all actual positive instances. High recall means the model captures most bankruptcies.
    
    - **F1 Score**: The harmonic mean of precision and recall. It's a good metric when you need to balance precision and recall.
    
    - **AUC**: Area Under the ROC Curve. Measures the model's ability to distinguish between classes. Higher values indicate better performance.
    """)
    
    # Class imbalance information
    st.markdown("### Class Imbalance")
    st.info("""
    **Note on Class Imbalance**: The dataset has a significant class imbalance with many more 'alive' companies than 'bankrupt' ones. 
    This imbalance affects metrics like accuracy, which can be high even when the model performs poorly on the minority class.
    
    For bankruptcy prediction, recall is particularly important as the cost of missing a bankruptcy (false negative) is typically higher 
    than incorrectly predicting bankruptcy (false positive).
    """)

elif selected_page == "ROC Curves":
    st.markdown('<p class="sub-header">ROC Curve Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### What are ROC Curves?
    
    ROC (Receiver Operating Characteristic) curves plot the True Positive Rate against the False Positive Rate at different classification thresholds. They show the tradeoff between sensitivity (recall) and specificity.
    
    - A model with perfect classification would have an AUC (Area Under the Curve) of 1.0
    - A model with no discrimination ability would have an AUC of 0.5 (equivalent to random guessing)
    """)
    
    # Select models to display
    model_options = list(metrics.keys())
    selected_models = st.multiselect(
        "Select models to compare", 
        options=model_options,
        default=model_options[:3]  # Default to first 3 models
    )
    
    if selected_models:
        # Plot ROC curves
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Add diagonal reference line (random classifier)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.8, label='Random')
        
        # Color map for different models
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Plot each selected model
        for i, model in enumerate(selected_models):
            fpr = roc_curves[model]['fpr']
            tpr = roc_curves[model]['tpr']
            auc = roc_curves[model]['auc']
            
            ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)], 
                    label=f'{model} (AUC = {auc:.3f})')
        
        # Set labels and title
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        
        # Add legend
        ax.legend(loc='lower right')
        
        # Set limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        # Add grid
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Show individual model ROC curve
    st.markdown("### Individual Model ROC Curve")
    
    # Select a single model for detailed view
    single_model = st.selectbox("Select a model", options=model_options)
    
    # Plot individual ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.8, label='Random')
    
    # Plot ROC curve
    fpr = roc_curves[single_model]['fpr']
    tpr = roc_curves[single_model]['tpr']
    auc = roc_curves[single_model]['auc']
    
    ax.plot(fpr, tpr, lw=2, color='#395c40', label=f'ROC curve (AUC = {auc:.3f})')
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{single_model} ROC Curve')
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Set limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add grid
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add explanation
    st.markdown("""
    ### Interpreting ROC Curves
    
    - **AUC (Area Under the Curve)**: The primary metric derived from the ROC curve. Higher values indicate better discriminative ability.
    
    - **Thresholds**: Each point on the ROC curve represents a different classification threshold. Moving along the curve shows the tradeoff between:
      - True Positive Rate (sensitivity/recall)
      - False Positive Rate (1 - specificity)
    
    - **Optimal Threshold**: The optimal threshold depends on the relative costs of false positives vs. false negatives. In bankruptcy prediction:
      - If missing a bankruptcy is very costly, choose a threshold with higher recall (upper right)
      - If falsely flagging healthy companies is costly, choose a threshold with higher specificity (lower left)
    """)

elif selected_page == "Feature Importance":
    st.markdown('<p class="sub-header">Feature Importance Analysis</p>', unsafe_allow_html=True)
    
    # Select model for feature importance
    model_options = ["Decision Tree", "Gradient Boosting", "Random Forest", "Logistic Regression", "KNN", "SVM"]
    selected_model = st.selectbox("Select model", model_options)
    
    # Get feature importances for selected model
    importances = pd.Series(feature_importances[selected_model]).sort_values(ascending=False)
    
    # Number of features to display
    n_features = st.slider("Number of top features to display", 5, len(feature_names), 10)
    
    # Plot feature importances
    st.markdown(f"### Top {n_features} Features for {selected_model}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get subset of data for visualization
    top_features = importances.head(n_features)
    
    # Find max value to set consistent x-limit
    max_value = importances.max()
    
    # Create horizontal bar chart
    bars = ax.barh(top_features.index[::-1], top_features.values[::-1], color='#395c40')
    
    # Add values to bars with consistent positioning
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max_value * 0.05,  # Position text at a fixed offset (5% of max value)
            bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            va='center',
            ha='left'  # Left-align all text for consistency
        )
    
    # Set consistent x-axis limit
    ax.set_xlim(0, max_value * 1.25)  # Add 25% padding
    
    ax.set_xlabel('Importance')
    ax.set_title(f'{selected_model} Feature Importance')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Compare top features across models
    st.markdown("### Top 5 Features Across Models")
    
    # Create comparison DataFrame - using main four models with more meaningful importances
    comparison_models = ["Decision Tree", "Gradient Boosting", "Random Forest", "Logistic Regression"]
    comparison_df = pd.DataFrame(index=range(1, 6), columns=comparison_models)
    
    for model in comparison_models:
        model_importances = pd.Series(feature_importances[model]).sort_values(ascending=False)
        for i in range(5):
            if i < len(model_importances):
                comparison_df.loc[i+1, model] = model_importances.index[i]
    
    st.dataframe(comparison_df)
    
    # Add explanation
    st.markdown("""
    ### Interpreting Feature Importance
    
    Different models calculate feature importance in different ways:
    
    - **Decision Tree**: Based on the total reduction of impurity (e.g., Gini impurity) contributed by each feature
    
    - **Gradient Boosting & Random Forest**: Based on the average reduction in impurity across all trees
    
    - **Logistic Regression**: Based on the absolute values of the coefficients (larger coefficient = more important)
    
    - **KNN & SVM**: Based on permutation importance - how much the model's performance decreases when the feature values are randomly shuffled
    
    For bankruptcy prediction, important features typically include financial ratios and indicators that capture:
    - Profitability (Net Income, EBITDA)
    - Leverage (Debt ratios)
    - Liquidity (Current Assets, Cash Flow)
    - Activity (Asset Turnover)
    """)

elif selected_page == "Confusion Matrices":
    st.markdown('<p class="sub-header">Confusion Matrix Analysis</p>', unsafe_allow_html=True)
    
    # Select model for confusion matrix
    model_options = list(metrics.keys())
    selected_model = st.selectbox("Select model", model_options)
    
    # Get confusion matrix for selected model
    cm = metrics[selected_model]['confusion_matrix']
    
    # Calculate additional metrics
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display confusion matrix in a simpler way without using matplotlib's colormap
    st.markdown(f"### {selected_model} Confusion Matrix")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Use st.dataframe for confusion matrix display
        cm_df = pd.DataFrame(
            cm,
            index=['Actual Alive', 'Actual Bankrupt'],
            columns=['Predicted Alive', 'Predicted Bankrupt']
        )
        
        st.dataframe(cm_df)
        
        # Create a simplified visualization
        st.markdown("### Visual Representation")
        
        # Convert to percentages for better comparison
        cm_pct = np.zeros((2, 2))
        cm_pct[0, 0] = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0  # TN as % of actual alive
        cm_pct[0, 1] = 100 * fp / (tn + fp) if (tn + fp) > 0 else 0  # FP as % of actual alive
        cm_pct[1, 0] = 100 * fn / (fn + tp) if (fn + tp) > 0 else 0  # FN as % of actual bankrupt
        cm_pct[1, 1] = 100 * tp / (fn + tp) if (fn + tp) > 0 else 0  # TP as % of actual bankrupt
        
        # Create a simple visualization using colored text
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
        .tn {{
            background-color: rgba(57, 92, 64, 0.8);
        }}
        .fp {{
            background-color: rgba(166, 54, 3, 0.8);
        }}
        .fn {{
            background-color: rgba(166, 54, 3, 0.8);
        }}
        .tp {{
            background-color: rgba(57, 92, 64, 0.8);
        }}
        </style>
        <div class="box-container">
            <div class="cm-box tn">
                True Negative<br>
                {tn} instances<br>
                ({cm_pct[0, 0]:.1f}% of actual alive)
            </div>
            <div class="cm-box fp">
                False Positive<br>
                {fp} instances<br>
                ({cm_pct[0, 1]:.1f}% of actual alive)
            </div>
            <div class="cm-box fn">
                False Negative<br>
                {fn} instances<br>
                ({cm_pct[1, 0]:.1f}% of actual bankrupt)
            </div>
            <div class="cm-box tp">
                True Positive<br>
                {tp} instances<br>
                ({cm_pct[1, 1]:.1f}% of actual bankrupt)
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Metrics")
        st.markdown(f"""
        - **True Negatives (TN)**: {tn}
        - **False Positives (FP)**: {fp}
        - **False Negatives (FN)**: {fn}
        - **True Positives (TP)**: {tp}
        
        - **Accuracy**: {accuracy:.4f}
        - **Precision**: {precision:.4f}
        - **Recall**: {recall:.4f}
        - **F1 Score**: {f1:.4f}
        """)
    
    # Add explanation
    st.markdown("""
    ### Understanding the Confusion Matrix
    
    - **True Negatives (TN)**: Companies correctly predicted as alive
    - **False Positives (FP)**: Companies incorrectly predicted as bankrupt
    - **False Negatives (FN)**: Bankrupt companies incorrectly predicted as alive
    - **True Positives (TP)**: Companies correctly predicted as bankrupt
    
    In bankruptcy prediction:
    - **False Negatives** are particularly costly (missed bankruptcies)
    - **False Positives** can also be problematic (incorrectly flagging healthy companies)
    
    The ideal model would maximize True Positives while minimizing False Negatives.
    """)
    
    # Compare confusion matrices across models
    st.markdown("### Bankruptcy Detection Comparison")
    
    # Calculate metrics for all models
    comparison_df = pd.DataFrame(
        index=model_options,
        columns=["True Positives", "False Negatives", "Detection Rate (%)", "False Alarm Rate (%)"]
    )
    
    for model in model_options:
        model_cm = metrics[model]['confusion_matrix']
        tn, fp = model_cm[0]
        fn, tp = model_cm[1]
        
        comparison_df.loc[model, "True Positives"] = tp
        comparison_df.loc[model, "False Negatives"] = fn
        comparison_df.loc[model, "Detection Rate (%)"] = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        comparison_df.loc[model, "False Alarm Rate (%)"] = 100 * fp / (tn + fp) if (tn + fp) > 0 else 0
    
    # Sort by detection rate
    comparison_df = comparison_df.sort_values("Detection Rate (%)", ascending=False)
    
    st.dataframe(comparison_df)
    
    # Add interpretation
    st.info("""
    **Note**: A higher detection rate (recall) means the model identifies more bankruptcies, but this often comes
    at the cost of more false alarms. The Decision Tree has the highest bankruptcy detection rate but also the highest
    false alarm rate, while SVM has the lowest false alarm rate but also the lowest detection rate.
    """)

elif selected_page == "Z-Score Analysis":
    st.markdown('<p class="sub-header">Altman Z-Score Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### What is the Altman Z-Score?
    
    The Altman Z-Score is a financial formula developed by Edward Altman in 1968 to predict the probability of a company going bankrupt. 
    It combines multiple financial ratios into a single score that helps assess a company's financial health.
    
    **Original Z-score formula:**
    
    Z = 1.2*T1 + 1.4*T2 + 3.3*T3 + 0.6*T4 + 0.99*T5
    
    Where:
    - T1 = Working Capital / Total Assets = (Current Assets - Current Liabilities) / Total Assets
    - T2 = Retained Earnings / Total Assets
    - T3 = Earnings Before Interest and Taxes / Total Assets
    - T4 = Market Value of Equity / Book Value of Total Liabilities
    - T5 = Sales / Total Assets
    
    ### Interpretation
    
    - Z > 2.99: "Safe" Zone - Company is in good financial health
    - 1.8 < Z < 2.99: "Grey" Zone - Some financial concerns exist
    - Z < 1.80: "Distress" Zone - High risk of bankruptcy
    """)
    
    # Check if data is loaded
    if st.session_state.get('data_loaded', False) and not data.empty:
        # Check for required columns
        required_cols = ['Current Assets', 'Total Current Liabilities', 'Retained Earnings', 
                         'Total Assets', 'EBIT', 'Market Value', 'Total Liabilities', 'Net Sales']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            st.error(f"Cannot calculate Z-Score: Missing required columns: {', '.join(missing_cols)}")
            st.warning("Please ensure your data file includes all necessary financial metrics.")
        else:
            # Calculate Z-Score
            if 'zscore_df' not in st.session_state:
                # Use actual data from the loaded CSV
                zscore_df = calculate_zscore(data)
                if not zscore_df.empty:
                    # Add actual bankruptcy status if available in the data
                    if 'Bankrupt' in data.columns:
                        zscore_df['Actual Status'] = data['Bankrupt']
                    st.session_state['zscore_df'] = zscore_df
            
            # Get Z-Score dataframe from session state
            zscore_df = st.session_state.get('zscore_df', pd.DataFrame())
            
            if not zscore_df.empty:
                # Display Z-Score distribution
                st.markdown("### Z-Score Distribution")
                
                # Check if actual bankruptcy status is available
                has_actual_status = 'Actual Status' in zscore_df.columns
                
                # Create histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if has_actual_status:
                    # Split by actual bankruptcy status
                    z_bankrupt = zscore_df[zscore_df['Actual Status'] == 1]['Z-Score']
                    z_healthy = zscore_df[zscore_df['Actual Status'] == 0]['Z-Score']
                    
                    # Plot histograms
                    ax.hist(z_healthy, bins=30, alpha=0.7, label='Healthy Companies', color='#395c40')
                    ax.hist(z_bankrupt, bins=30, alpha=0.7, label='Bankrupt Companies', color='#a63603')
                else:
                    # Plot single histogram without status distinction
                    ax.hist(zscore_df['Z-Score'], bins=30, alpha=0.7, color='#395c40')
                
                # Add vertical lines for Z-Score zones
                ax.axvline(x=1.8, color='red', linestyle='--', alpha=0.7, label='Z=1.8 (Distress Threshold)')
                ax.axvline(x=2.99, color='green', linestyle='--', alpha=0.7, label='Z=2.99 (Safe Threshold)')
                
                ax.set_xlabel('Z-Score')
                ax.set_ylabel('Number of Companies')
                ax.set_title('Z-Score Distribution')
                ax.legend()
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Calculate Z-Score performance metrics
                if has_actual_status:
                    z_pred = zscore_df['Z-Score Prediction'].values
                    z_actual = zscore_df['Actual Status'].values
                    
                    # Calculate metrics
                    z_accuracy = (z_pred == z_actual).mean()
                    z_precision = (z_pred & z_actual).sum() / z_pred.sum() if z_pred.sum() > 0 else 0
                    z_recall = (z_pred & z_actual).sum() / z_actual.sum() if z_actual.sum() > 0 else 0
                    z_f1 = 2 * z_precision * z_recall / (z_precision + z_recall) if (z_precision + z_recall) > 0 else 0
                    
                    # Calculate Z-Score confusion matrix
                    z_tn = ((z_pred == 0) & (z_actual == 0)).sum()
                    z_fp = ((z_pred == 1) & (z_actual == 0)).sum()
                    z_fn = ((z_pred == 0) & (z_actual == 1)).sum()
                    z_tp = ((z_pred == 1) & (z_actual == 1)).sum()
                    
                    # Compare Z-Score with ML models
                    st.markdown("### Z-Score vs. Machine Learning Models")
                    
                    # Create comparison DataFrame
                    comparison_data = {
                        'Model': ['Altman Z-Score'] + list(metrics.keys()),
                        'Accuracy': [z_accuracy] + [metrics[model]['accuracy'] for model in metrics],
                        'Precision': [z_precision] + [metrics[model]['precision'] for model in metrics],
                        'Recall': [z_recall] + [metrics[model]['recall'] for model in metrics],
                        'F1 Score': [z_f1] + [metrics[model]['f1'] for model in metrics]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data).set_index('Model')
                    
                    # Display comparison
                    st.dataframe(comparison_df.style.highlight_max(axis=0))
                    
                    # Z-Score confusion matrix
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("### Z-Score Confusion Matrix")
                        z_cm_df = pd.DataFrame(
                            [[z_tn, z_fp], [z_fn, z_tp]],
                            index=['Actual Alive', 'Actual Bankrupt'],
                            columns=['Predicted Alive', 'Predicted Bankrupt']
                        )
                        st.dataframe(z_cm_df)
                        
                        # Visual representation of confusion matrix
                        cm_pct = np.zeros((2, 2))
                        cm_pct[0, 0] = 100 * z_tn / (z_tn + z_fp) if (z_tn + z_fp) > 0 else 0
                        cm_pct[0, 1] = 100 * z_fp / (z_tn + z_fp) if (z_tn + z_fp) > 0 else 0
                        cm_pct[1, 0] = 100 * z_fn / (z_fn + z_tp) if (z_fn + z_tp) > 0 else 0
                        cm_pct[1, 1] = 100 * z_tp / (z_fn + z_tp) if (z_fn + z_tp) > 0 else 0
                        
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
                        .tn {{
                            background-color: rgba(57, 92, 64, 0.8);
                        }}
                        .fp {{
                            background-color: rgba(166, 54, 3, 0.8);
                        }}
                        .fn {{
                            background-color: rgba(166, 54, 3, 0.8);
                        }}
                        .tp {{
                            background-color: rgba(57, 92, 64, 0.8);
                        }}
                        </style>
                        <div class="box-container">
                            <div class="cm-box tn">
                                True Negative<br>
                                {z_tn} instances<br>
                                ({cm_pct[0, 0]:.1f}% of actual alive)
                            </div>
                            <div class="cm-box fp">
                                False Positive<br>
                                {z_fp} instances<br>
                                ({cm_pct[0, 1]:.1f}% of actual alive)
                            </div>
                            <div class="cm-box fn">
                                False Negative<br>
                                {z_fn} instances<br>
                                ({cm_pct[1, 0]:.1f}% of actual bankrupt)
                            </div>
                            <div class="cm-box tp">
                                True Positive<br>
                                {z_tp} instances<br>
                                ({cm_pct[1, 1]:.1f}% of actual bankrupt)
                            </div>
                        </div>
                        """
                        st.markdown(html, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### Z-Score Metrics")
                        st.markdown(f"""
                        - **True Negatives (TN)**: {z_tn}
                        - **False Positives (FP)**: {z_fp}
                        - **False Negatives (FN)**: {z_fn}
                        - **True Positives (TP)**: {z_tp}
                        
                        - **Accuracy**: {z_accuracy:.4f}
                        - **Precision**: {z_precision:.4f}
                        - **Recall**: {z_recall:.4f}
                        - **F1 Score**: {z_f1:.4f}
                        """)
                
                # Add financial insight
                st.markdown("### Financial Insights")
                st.markdown("""
                The Altman Z-Score is widely used in financial analysis for predicting bankruptcy risk. It combines multiple financial ratios 
                that measure profitability, leverage, liquidity, solvency, and activity into a single score.
                
                #### Comparing with Machine Learning Models:
                
                - **Interpretability**: Z-Score is easy to interpret and communicate to stakeholders
                - **Simplicity**: Simple linear combination of 5 financial ratios
                - **Historical validation**: Well-established method with decades of validation
                
                #### Limitations:
                
                - **Static weights**: Uses fixed coefficients that don't adapt to changing economic conditions
                - **Limited inputs**: Only uses 5 financial ratios, while ML models can incorporate more features
                - **No industry adjustment**: Same thresholds for all industries, unlike ML models that can learn industry-specific patterns
                """)
            else:
                st.warning("Could not calculate Z-Score with the available data. Please ensure the dataset contains the necessary financial metrics.")
    else:
        st.error("Please upload your data file. The data file should be named 'american_bankruptcy.csv' and located in the 'data/' directory.")
        st.info("""
        The dataset should include the following financial metrics:
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
<div style="text-align: center; color: #888888; font-size: 0.8em;">
Bankruptcy Prediction Dashboard | Created with Streamlit | Data Analysis Based on Financial Metrics
</div>
""", unsafe_allow_html=True)
