import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import io
import base64

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

# Define feature names and model results based on your analysis
feature_names = [
    "Current Assets", "Cost of Goods Sold", "D&A", "EBITDA",
    "Inventory", "Net Income", "Total Receivables", "Market Value",
    "Net Sales", "Total Assets", "Total Long-term Debt", "EBIT",
    "Gross Profit", "Total Current Liabilities", "Retained Earnings",
    "Total Revenue", "Total Liabilities", "Total Operating Expenses"
]

# Define metrics from your analysis
metrics = {
    'Decision Tree': {
        'accuracy': 0.8925,
        'precision': 0.0589,
        'recall': 0.2404,
        'f1': 0.0947,
        'auc': 0.6171,  # Using ROC AUC value from the plots
        'confusion_matrix': [[10893, 1102], [218, 69]]
    },
    'Gradient Boosting': {
        'accuracy': 0.9761,
        'precision': 0.3846,
        'recall': 0.0348,
        'f1': 0.0639,
        'auc': 0.9183,
        'confusion_matrix': [[11979, 16], [277, 10]]
    },
    'Random Forest': {
        'accuracy': 0.9759,
        'precision': 0.3200,
        'recall': 0.0279,
        'f1': 0.0513,
        'auc': 0.9040,
        'confusion_matrix': [[11978, 17], [279, 8]]
    },
    'Logistic Regression': {
        'accuracy': 0.9752,
        'precision': 0.3125,
        'recall': 0.0523,
        'f1': 0.0896,
        'auc': 0.8773,
        'confusion_matrix': [[11962, 33], [272, 15]]
    },
    'SVM': {
        'accuracy': 0.9765,
        'precision': 0.3333,
        'recall': 0.0070,
        'f1': 0.0137,
        'auc': 0.8635,
        'confusion_matrix': [[11991, 4], [285, 2]]
    },
    'KNN': {
        'accuracy': 0.9589,
        'precision': 0.1414,
        'recall': 0.1498,
        'f1': 0.1455,
        'auc': 0.6324,
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
        'Current Assets': 0.05991,
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
        'Total Receivables': 0.04075,
        'Net Income': 0.019487,
        'D&A': 0.006214,
        'Net Sales': 0.001644,
        'Total Revenue': 0.001644
    }
}

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Overview", "Model Comparison", "Feature Importance", "Confusion Matrices"]
selected_page = st.sidebar.radio("Go to", pages)

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

elif selected_page == "Feature Importance":
    st.markdown('<p class="sub-header">Feature Importance Analysis</p>', unsafe_allow_html=True)
    
    # Select model for feature importance
    model_options = ["Decision Tree", "Gradient Boosting", "Random Forest", "Logistic Regression"]
    selected_model = st.selectbox("Select model", model_options)
    
    # Get feature importances for selected model
    importances = pd.Series(feature_importances[selected_model]).sort_values(ascending=False)
    
    # Number of features to display
    n_features = st.slider("Number of top features to display", 5, len(feature_names), 10)
    
    # Plot feature importances
    st.markdown(f"### Top {n_features} Features for {selected_model}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(importances.head(n_features).index[::-1], importances.head(n_features).values[::-1], color='#395c40')
    
    # Add values to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            va='center'
        )
    
    ax.set_xlabel('Importance')
    ax.set_title(f'{selected_model} Feature Importance')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Display feature importance table
    st.markdown("### Complete Feature Ranking")
    st.dataframe(importances)
    
    # Compare top features across models
    st.markdown("### Top 5 Features Across Models")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(index=range(1, 6), columns=model_options)
    
    for model in model_options:
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
    
    # Display confusion matrix
    st.markdown(f"### {selected_model} Confusion Matrix")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Add labels
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        ax.set_title(f'{selected_model} Confusion Matrix', fontsize=14)
        
        # Add class labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Alive', 'Bankrupt'])
        ax.set_yticklabels(['Alive', 'Bankrupt'])
        
        # Add colorbar
        plt.colorbar(im)
        
        # Add text annotations - FIXED CODE HERE
        # Calculate threshold for text color
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i][j]}',
                      ha="center", va="center",
                      color="white" if cm[i][j] > thresh else "black",
                      fontsize=12)
        
        plt.tight_layout()
        st.pyplot(fig)
    
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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888888; font-size: 0.8em;">
Bankruptcy Prediction Dashboard | Created with Streamlit | Data Analysis Based on Financial Metrics
</div>
""", unsafe_allow_html=True)
