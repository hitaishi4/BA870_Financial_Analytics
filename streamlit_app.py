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
from sklearn.inspection import permutation_importance
import io
import base64
import time

# Set page configuration
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

# Define helper functions that were originally in separate files
# ===============================================================

# 1. VISUALIZATION FUNCTIONS
# ===============================================================

def plot_roc_curves(all_probabilities, y_test, model_names):
    """Plot ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # List to store AUC values for legend
    auc_values = []
    
    # Plot ROC curve for each model
    for i, (y_proba, name) in enumerate(zip(all_probabilities, model_names)):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)
        
        # Different line styles for better visibility
        line_styles = ['-', '--', '-.', ':', '-', '--']
        # Different colors for each model
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        ax.plot(
            fpr, tpr, 
            label=f'{name} (AUC = {roc_auc:.3f})',
            linestyle=line_styles[i % len(line_styles)],
            color=colors[i % len(colors)],
            linewidth=2
        )
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    
    # Set plot attributes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_feature_importances(importances, title=None, n_features=None):
    """Plot feature importances."""
    # Sort feature importances
    importances = importances.sort_values(ascending=False)
    
    # Limit to top n features if specified
    if n_features is not None and n_features < len(importances):
        importances = importances.iloc[:n_features]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    bars = ax.barh(
        importances.index[::-1], 
        importances.values[::-1],
        color='#395c40',
        alpha=0.8
    )
    
    # Add values to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + width * 0.02, 
            bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            ha='left',
            va='center'
        )
    
    # Set plot attributes
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title if title else 'Feature Importance', fontsize=16)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrices(y_true, y_pred, title=None):
    """Plot confusion matrix."""
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title(title if title else 'Confusion Matrix', fontsize=16)
    
    # Add class labels
    classes = ['Alive', 'Bankrupt']
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14
            )
    
    plt.tight_layout()
    return fig

def plot_model_comparison(metrics_df, selected_metrics):
    """Plot comparison of model performance for selected metrics."""
    # Check if selected metrics exist in the DataFrame
    for metric in selected_metrics:
        if metric not in metrics_df.columns:
            raise ValueError(f"Metric '{metric}' not found in metrics DataFrame")
    
    # Number of metrics to plot
    n_metrics = len(selected_metrics)
    
    # Create figure
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(selected_metrics):
        # Sort models by metric value
        sorted_df = metrics_df.sort_values(metric, ascending=False)
        
        # Create bar chart
        bars = axes[i].bar(
            sorted_df.index,
            sorted_df[metric],
            color='#395c40',
            alpha=0.8
        )
        
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
        
        # Set plot attributes
        axes[i].set_title(metric, fontsize=14)
        axes[i].set_ylim(0, sorted_df[metric].max() * 1.15)
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].set_xticklabels(sorted_df.index, rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# 2. DATA LOADING FUNCTIONS
# ===============================================================

def load_data(file_path=None):
    """Load the bankruptcy dataset or generate synthetic data."""
    if file_path:
        try:
            # Try to load the CSV file
            df = pd.read_csv(file_path)
            
            # Basic validation that this is the expected dataset
            required_columns = ['status_label']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
                    
            # Create bankruptcy indicator from status_label
            if 'Bankruptcy' not in df.columns:
                df['Bankruptcy'] = df['status_label'].map({'failed': 1, 'alive': 0})
                
            # Rename columns if X1, X2, etc. are present
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
            
            # Check if we need to rename columns (if X1, X2, etc. are present)
            needs_renaming = any(col in df.columns for col in rename_map.keys())
            if needs_renaming:
                df = df.rename(columns=rename_map)
            
            return df
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.info("Generating synthetic data instead.")
            return generate_sample_data()
    else:
        # Generate synthetic data if no file provided
        return generate_sample_data()

def generate_sample_data(n_samples=1000, bankruptcy_rate=0.05):
    """Generate synthetic bankruptcy data for testing or demonstration."""
    np.random.seed(42)
    
    # Feature names
    feature_names = [
        "Current Assets", "Cost of Goods Sold", "D&A", "EBITDA",
        "Inventory", "Net Income", "Total Receivables", "Market Value",
        "Net Sales", "Total Assets", "Total Long-term Debt", "EBIT",
        "Gross Profit", "Total Current Liabilities", "Retained Earnings",
        "Total Revenue", "Total Liabilities", "Total Operating Expenses"
    ]
    
    # Generate random data for non-bankrupt companies
    n_alive = int(n_samples * (1 - bankruptcy_rate))
    X_alive = np.random.normal(loc=5.0, scale=2.0, size=(n_alive, len(feature_names)))
    
    # Generate random data for bankrupt companies
    # Bankrupt companies tend to have lower values for positive metrics and higher for negative ones
    n_bankrupt = n_samples - n_alive
    X_bankrupt = np.random.normal(loc=3.0, scale=2.5, size=(n_bankrupt, len(feature_names)))
    
    # For bankrupt companies, adjust certain features to reflect financial distress
    # Net Income, Retained Earnings, and EBITDA tend to be lower for bankrupt companies
    income_idx = feature_names.index("Net Income")
    retained_idx = feature_names.index("Retained Earnings")
    ebitda_idx = feature_names.index("EBITDA")
    debt_idx = feature_names.index("Total Long-term Debt")
    
    # Adjust values to be more reflective of bankruptcy
    X_bankrupt[:, income_idx] = np.random.normal(loc=-1.0, scale=2.0, size=n_bankrupt)
    X_bankrupt[:, retained_idx] = np.random.normal(loc=0.5, scale=1.5, size=n_bankrupt)
    X_bankrupt[:, ebitda_idx] = np.random.normal(loc=0.8, scale=1.2, size=n_bankrupt)
    X_bankrupt[:, debt_idx] = np.random.normal(loc=7.0, scale=2.0, size=n_bankrupt)
    
    # Combine datasets
    X = np.vstack([X_alive, X_bankrupt])
    y = np.hstack([np.zeros(n_alive), np.ones(n_bankrupt)])
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add bankruptcy indicator
    df['Bankruptcy'] = y
    
    # Add status_label
    df['status_label'] = df['Bankruptcy'].map({0: 'alive', 1: 'failed'})
    
    # Add year column (spread across 1999-2018)
    years = np.random.choice(range(1999, 2019), n_samples)
    df['year'] = years
    
    # Ensure bankrupt companies are mostly in testing range (2015-2018)
    bankrupt_indices = df[df['Bankruptcy'] == 1].index
    alive_indices = df[df['Bankruptcy'] == 0].index
    
    # Assign most bankrupt companies to testing years
    n_test_bankrupt = int(n_bankrupt * 0.7)
    test_bankrupt_indices = np.random.choice(bankrupt_indices, n_test_bankrupt, replace=False)
    df.loc[test_bankrupt_indices, 'year'] = np.random.choice(range(2015, 2019), n_test_bankrupt)
    
    # Assign remaining bankrupt companies to training years
    train_bankrupt_indices = np.setdiff1d(bankrupt_indices, test_bankrupt_indices)
    df.loc[train_bankrupt_indices, 'year'] = np.random.choice(range(1999, 2012), len(train_bankrupt_indices))
    
    # Assign alive companies to both ranges
    n_test_alive = int(n_alive * 0.3)
    test_alive_indices = np.random.choice(alive_indices, n_test_alive, replace=False)
    df.loc[test_alive_indices, 'year'] = np.random.choice(range(2015, 2019), n_test_alive)
    
    train_alive_indices = np.setdiff1d(alive_indices, test_alive_indices)
    df.loc[train_alive_indices, 'year'] = np.random.choice(range(1999, 2012), len(train_alive_indices))
    
    return df

def preprocess_data(df):
    """Preprocess the bankruptcy dataset for model training and evaluation."""
    try:
        # Ensure the Bankruptcy column exists
        if 'Bankruptcy' not in df.columns:
            if 'status_label' in df.columns:
                df['Bankruptcy'] = df['status_label'].map({'failed': 1, 'alive': 0})
            else:
                raise ValueError("Missing both 'Bankruptcy' and 'status_label' columns")
        
        # Get feature names
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
        
        features = list(rename_map.values())
        
        # Make sure all expected features are in the dataframe
        # If not, create them with default values
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0.0
        
        # Handle missing values (if any)
        for feature in features:
            if df[feature].isnull().any():
                # Fill missing values with mean
                df[feature] = df[feature].fillna(df[feature].mean())
        
        # Split into train/test sets based on year (as in the original code)
        if 'year' in df.columns:
            # Use the same year ranges as in the original code
            train = df[(df.year >= 1999) & (df.year <= 2011)]
            test = df[(df.year >= 2015) & (df.year <= 2018)]
            
            # Check if we have enough data in both sets
            if len(train) < 10 or len(test) < 10:
                # Fall back to random split if we don't have enough data
                train = df.sample(frac=0.7, random_state=42)
                test = df.drop(train.index)
        else:
            # If year column is missing, split randomly
            train = df.sample(frac=0.7, random_state=42)
            test = df.drop(train.index)
        
        # Create X and y for training and testing
        X_train, y_train = train[features], train['Bankruptcy']
        X_test, y_test = test[features], test['Bankruptcy']
        
        # Handle any remaining NaN values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Convert to float to avoid any data type issues
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        
        return X_train, X_test, y_train, y_test, features
    
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None, None, None, None, None

# 3. MODEL TRAINING FUNCTIONS
# ===============================================================

def train_models(X_train, y_train):
    """Train multiple bankruptcy prediction models."""
    models = []
    
    # Decision Tree
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    models.append(dt_clf)
    
    # Gradient Boosting
    gb_clf = GradientBoostingClassifier(random_state=42)
    gb_clf.fit(X_train, y_train)
    models.append(gb_clf)
    
    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    models.append(rf_clf)
    
    # Logistic Regression (with standardization)
    logreg_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(solver='liblinear', random_state=42))
    ])
    logreg_pipeline.fit(X_train, y_train)
    models.append(logreg_pipeline)
    
    # Support Vector Machine (with standardization)
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True, random_state=42))
    ])
    svm_pipeline.fit(X_train, y_train)
    models.append(svm_pipeline)
    
    # K-Nearest Neighbors (with standardization)
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    knn_pipeline.fit(X_train, y_train)
    models.append(knn_pipeline)
    
    return models

def evaluate_models(models, X_test, y_test, feature_names=None):
    """Evaluate multiple bankruptcy prediction models."""
    model_names = [
        "Decision Tree",
        "Gradient Boosting",
        "Random Forest",
        "Logistic Regression",
        "SVM",
        "KNN"
    ]
    
    results = {}
    all_predictions = []
    all_probabilities = []
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Store predictions and probabilities
        all_predictions.append(y_pred)
        all_probabilities.append(y_proba)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            "predictions": y_pred,
            "probabilities": y_proba,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": roc_auc,
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr
        }
        
        # Add feature importances if available
        if feature_names is not None:
            if name == "Decision Tree":
                results[name]["feature_importance"] = pd.Series(
                    model.feature_importances_, index=feature_names
                ).sort_values(ascending=False)
            
            elif name == "Gradient Boosting":
                results[name]["feature_importance"] = pd.Series(
                    model.feature_importances_, index=feature_names
                ).sort_values(ascending=False)
            
            elif name == "Random Forest":
                results[name]["feature_importance"] = pd.Series(
                    model.feature_importances_, index=feature_names
                ).sort_values(ascending=False)
            
            elif name == "Logistic Regression":
                results[name]["feature_importance"] = pd.Series(
                    np.abs(model.named_steps['logreg'].coef_[0]), index=feature_names
                ).sort_values(ascending=False)
    
    return results, all_predictions, all_probabilities

# 4. UTILITY FUNCTIONS
# ===============================================================

def get_model_names():
    """Get list of model names used in the dashboard."""
    return [
        "Decision Tree",
        "Gradient Boosting",
        "Random Forest",
        "Logistic Regression",
        "SVM",
        "KNN"
    ]

def get_metrics_df(results):
    """Create a DataFrame of performance metrics for all models."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    metrics_df = pd.DataFrame(index=results.keys(), columns=metrics)
    
    for model_name, model_results in results.items():
        metrics_df.loc[model_name, 'Accuracy'] = model_results['accuracy']
        metrics_df.loc[model_name, 'Precision'] = model_results['precision']
        metrics_df.loc[model_name, 'Recall'] = model_results['recall']
        metrics_df.loc[model_name, 'F1 Score'] = model_results['f1']
        metrics_df.loc[model_name, 'AUC'] = model_results['auc']
    
    return metrics_df

def get_best_model(metrics_df, metric='AUC'):
    """Get the best performing model based on a specific metric."""
    return metrics_df[metric].idxmax()

def format_confusion_matrix(cm):
    """Format confusion matrix for display."""
    return pd.DataFrame(
        cm,
        index=['Actual Alive', 'Actual Bankrupt'],
        columns=['Predicted Alive', 'Predicted Bankrupt']
    )

# MAIN APPLICATION
# ===============================================================

def main():
    # App title and introduction
    st.markdown('<p class="main-header">Bankruptcy Prediction Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard presents a comprehensive analysis of bankruptcy prediction models using financial data 
    from American companies. The analysis includes model performance comparisons, feature importance analysis, 
    and detailed evaluation metrics.
    """)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    pages = ["Overview", "Data Exploration", "Model Comparison", "Feature Importance", "Detailed Analysis"]
    selected_page = st.sidebar.radio("Go to", pages)
    
    # Sidebar for data upload (optional)
    st.sidebar.title("Data Options")
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if not use_sample_data:
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.sidebar.warning("Please upload a data file or use sample data")
            df = None
    else:
        with st.spinner("Loading sample data..."):
            try:
                # Use sample data
                st.sidebar.info("Using generated sample data")
                df = generate_sample_data()
                st.sidebar.success("Sample data loaded!")
            except Exception as e:
                st.sidebar.error(f"Error loading sample data: {e}")
                df = None
    
    # Keep track of trained models and results
    if 'trained_models' not in st.session_state:
        st.session_state['trained_models'] = None
        st.session_state['results'] = None
        st.session_state['X_train'] = None
        st.session_state['X_test'] = None
        st.session_state['y_train'] = None
        st.session_state['y_test'] = None
        st.session_state['feature_names'] = None
        st.session_state['all_predictions'] = None
        st.session_state['all_probabilities'] = None
    
    # Process data if available
    if df is not None and (st.session_state['trained_models'] is None or not use_sample_data):
        with st.spinner("Processing data and training models..."):
            # Preprocess data
            X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
            
            if X_train is not None:
                # Store in session state
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = feature_names
                
                # Train models
                models = train_models(X_train, y_train)
                st.session_state['trained_models'] = models
                
                # Evaluate models
                results, all_predictions, all_probabilities = evaluate_models(
                    models, X_test, y_test, feature_names
                )
                st.session_state['results'] = results
                st.session_state['all_predictions'] = all_predictions
                st.session_state['all_probabilities'] = all_probabilities
    
    # Render the selected page
    if selected_page == "Overview":
        render_overview()
    elif selected_page == "Data Exploration":
        render_data_exploration()
    elif selected_page == "Model Comparison":
        render_model_comparison()
    elif selected_page == "Feature Importance":
        render_feature_importance()
    elif selected_page == "Detailed Analysis":
        render_detailed_analysis()

def render_overview():
    st.markdown('<p class="sub-header">Overview</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Summary
    
    This dashboard presents a detailed analysis of bankruptcy prediction models using financial data
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
    
    # Display summary of results if available
    if st.session_state['results'] is not None:
        st.markdown('<p class="section-header">Performance Summary</p>', unsafe_allow_html=True)
        
        # Create metrics dataframe
        metrics_df = get_metrics_df(st.session_state['results'])
        
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
    
    # Navigation buttons
    st.markdown("### Explore the dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Explore Data"):
            st.session_state["page"] = "Data Exploration"
            st.experimental_rerun()
    with col2:
        if st.button("Compare Models"):
            st.session_state["page"] = "Model Comparison"
            st.experimental_rerun()
    with col3:
        if st.button("Analyze Features"):
            st.session_state["page"] = "Feature Importance"
            st.experimental_rerun()

def render_data_exploration():
    st.markdown('<p class="sub-header">Data Exploration</p>', unsafe_allow_html=True)
    
    if st.session_state['X_train'] is not None and st.session_state['y_train'] is not None:
        # Get data from session state
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        feature_names = st.session_state['feature_names']
        
        # Display dataset information
        st.markdown("### Dataset Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Training Set:**")
            st.write(f"- Samples: {X_train.shape[0]}")
            st.write(f"- Features: {X_train.shape[1]}")
            st.write(f"- Bankruptcy rate: {y_train.mean():.2%}")
        
        with col2:
            st.write(f"**Testing Set:**")
            st.write(f"- Samples: {X_test.shape[0]}")
            st.write(f"- Features: {X_test.shape[1]}")
            st.write(f"- Bankruptcy rate: {y_test.mean():.2%}")
        
        st.markdown("### Feature Distribution")
        
        # Feature selection
        selected_feature = st.selectbox("Select Feature", feature_names)
        
        # Plot feature distribution
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training data distribution
        train_data = pd.DataFrame({
            selected_feature: X_train[selected_feature],
            'Status': ['Bankrupt' if x == 1 else 'Alive' for x in y_train]
        })
        
        # Calculate means for the selected feature by status
        alive_mean = train_data[train_data['Status'] == 'Alive'][selected_feature].mean()
        bankrupt_mean = train_data[train_data['Status'] == 'Bankrupt'][selected_feature].mean()
        
        train_alive = train_data[train_data['Status'] == 'Alive'][selected_feature]
        train_bankrupt = train_data[train_data['Status'] == 'Bankrupt'][selected_feature]
        
        ax[0].hist([train_alive, train_bankrupt], bins=20, alpha=0.7, 
                  label=['Alive', 'Bankrupt'])
        ax[0].axvline(alive_mean, color='blue', linestyle='dashed', linewidth=1)
        ax[0].axvline(bankrupt_mean, color='orange', linestyle='dashed', linewidth=1)
        ax[0].set_title(f"{selected_feature} - Training Data")
        ax[0].legend()
        
        # Testing data distribution
        test_data = pd.DataFrame({
            selected_feature: X_test[selected_feature],
            'Status': ['Bankrupt' if x == 1 else 'Alive' for x in y_test]
        })
        
        test_alive = test_data[test_data['Status'] == 'Alive'][selected_feature]
        test_bankrupt = test_data[test_data['Status'] == 'Bankrupt'][selected_feature]
        
        ax[1].hist([test_alive, test_bankrupt], bins=20, alpha=0.7, 
                  label=['Alive', 'Bankrupt'])
        ax[1].set_title(f"{selected_feature} - Testing Data")
        ax[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Feature correlation analysis
        st.markdown("### Feature Correlation Analysis")
        
        # Calculate correlation matrix
        corr_matrix = X_train.corr()
        
        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='coolwarm')
        
        # Add feature names as ticks
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=90)
        ax.set_yticklabels(feature_names)
        
        # Add colorbar
        plt.colorbar(im)
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show top correlated features
        st.markdown("### Top Feature Correlations")
        
        # Get upper triangle of correlation matrix (excluding diagonal)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find top 5 correlations
        top_corr = upper.unstack().sort_values(ascending=False)[:5]
        
        for i, (pair, corr) in enumerate(top_corr.items()):
            st.write(f"{i+1}. **{pair[0]} & {pair[1]}**: {corr:.3f}")
    else:
        st.warning("No data available. Please load data or use sample data.")

def render_model_comparison():
    st.markdown('<p class="sub-header">Model Performance Comparison</p>', unsafe_allow_html=True)
    
    if st.session_state['results'] is not None:
        # Get results from session state
        results = st.session_state['results']
        all_predictions = st.session_state['all_predictions']
        all_probabilities = st.session_state['all_probabilities']
        y_test = st.session_state['y_test']
        
        # 1. Metrics comparison
        st.markdown("### Performance Metrics")
        
        metrics_df = get_metrics_df(results)
        st.dataframe(metrics_df.style.highlight_max(axis=0))
        
        # 2. Visualization options
        st.markdown("### Visualize Comparisons")
        
        metric_options = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
        selected_metrics = st.multiselect(
            "Select metrics to compare", 
            options=metric_options,
            default=["AUC", "F1 Score"]
        )
        
        if selected_metrics:
            # Plot selected metrics comparison
            fig = plot_model_comparison(metrics_df, selected_metrics)
            st.pyplot(fig)
        
        # 3. ROC Curves
        st.markdown("### ROC Curves")
        
        # Plot ROC curves for all models
        fig = plot_roc_curves(all_probabilities, y_test, get_model_names())
        st.pyplot(fig)
        
        # 4. Confusion Matrices
        st.markdown("### Confusion Matrices")
        
        # Select model for confusion matrix
        models = get_model_names()
        selected_model = st.selectbox("Select model", models)
        
        # Plot confusion matrix for selected model
        model_idx = models.index(selected_model)
        fig = plot_confusion_matrices(y_test, all_predictions[model_idx], selected_model)
        st.pyplot(fig)
        
        # 5. Classification Reports
        st.markdown("### Classification Report")
        
        # Generate classification report for selected model
        model_idx = models.index(selected_model)
        report = classification_report(
            y_test, 
            all_predictions[model_idx], 
            target_names=["Alive", "Bankrupt"],
            output_dict=True
        )
        
        # Convert report to DataFrame
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.style.highlight_max(axis=1))
    else:
        st.warning("No results available. Please load data and train models.")

def render_feature_importance():
    st.markdown('<p class="sub-header">Feature Importance Analysis</p>', unsafe_allow_html=True)
    
    if st.session_state['trained_models'] is not None:
        # Get data from session state
        models = st.session_state['trained_models']
        feature_names = st.session_state['feature_names']
        
        # Select model for feature importance
        model_options = ["Decision Tree", "Gradient Boosting", "Random Forest", "Logistic Regression"]
        selected_model = st.selectbox("Select model", model_options)
        
        # Get feature importances for selected model
        if selected_model == "Decision Tree":
            importances = models[0].feature_importances_
            title = "Decision Tree Feature Importance"
        elif selected_model == "Gradient Boosting":
            importances = models[1].feature_importances_
            title = "Gradient Boosting Feature Importance"
        elif selected_model == "Random Forest":
            importances = models[2].feature_importances_
            title = "Random Forest Feature Importance"
        elif selected_model == "Logistic Regression":
            importances = np.abs(models[3].named_steps['logreg'].coef_[0])
            title = "Logistic Regression Coefficient Magnitude"
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Number of top features to display
        n_features = st.slider("Number of features to display", 5, len(feature_names), 10)
        
        # Plot feature importances
        fig = plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'][:n_features][::-1], 
                importance_df['Importance'][:n_features][::-1])
        plt.xlabel('Importance')
        plt.title(title)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display feature importance table
        st.markdown("### Feature Importance Ranking")
        st.dataframe(importance_df)
        
        # Feature importance comparison across models
        st.markdown("### Compare Top Features Across Models")
        
        # Get top 5 features for each model
        dt_importances = pd.Series(
            models[0].feature_importances_, 
            index=feature_names
        ).sort_values(ascending=False).head(5)
        
        gb_importances = pd.Series(
            models[1].feature_importances_, 
            index=feature_names
        ).sort_values(ascending=False).head(5)
        
        rf_importances = pd.Series(
            models[2].feature_importances_, 
            index=feature_names
        ).sort_values(ascending=False).head(5)
        
        lr_importances = pd.Series(
            np.abs(models[3].named_steps['logreg'].coef_[0]), 
            index=feature_names
        ).sort_values(ascending=False).head(5)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Decision Tree': dt_importances.index,
            'DT Importance': dt_importances.values,
            'Gradient Boosting': gb_importances.index,
            'GB Importance': gb_importances.values,
            'Random Forest': rf_importances.index,
            'RF Importance': rf_importances.values,
            'Logistic Regression': lr_importances.index,
            'LR Importance': lr_importances.values
        })
        
        st.dataframe(comparison_df)
    else:
        st.warning("No models available. Please load data and train models.")

def render_detailed_analysis():
    st.markdown('<p class="sub-header">Detailed Model Analysis</p>', unsafe_allow_html=True)
    
    if st.session_state['results'] is not None:
        # Get data from session state
        models = st.session_state['trained_models']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        all_predictions = st.session_state['all_predictions']
        all_probabilities = st.session_state['all_probabilities']
        
        # Tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Model Deep Dive", "Error Analysis", "Sample Predictions"])
        
        with tab1:
            st.markdown("### Model Performance Deep Dive")
            
            # Select model for detailed analysis
            model_options = get_model_names()
            selected_model = st.selectbox("Select model for analysis", model_options, key="tab1_model")
            model_idx = model_options.index(selected_model)
            
            # Get predictions and probabilities for selected model
            y_pred = all_predictions[model_idx]
            y_proba = all_probabilities[model_idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Display confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(
                    cm, 
                    index=['Actual 0','Actual 1'], 
                    columns=['Pred 0','Pred 1']
                )
                st.markdown("#### Confusion Matrix")
                st.dataframe(cm_df)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                st.markdown("#### Performance Metrics")
                st.write(f"Accuracy : {accuracy:.4f}")
                st.write(f"Precision: {precision:.4f}")
                st.write(f"Recall   : {recall:.4f}")
                st.write(f"F1 Score : {f1:.4f}")
            
            with col2:
                # Plot ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr)
                ax.plot([0, 1], [0, 1], linestyle='--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab2:
            st.markdown("### Error Analysis")
            
            # Select model for error analysis
            model_options = get_model_names()
            selected_model = st.selectbox("Select model for error analysis", model_options, key="tab2_model")
            model_idx = model_options.index(selected_model)
            
            # Get predictions for selected model
            y_pred = all_predictions[model_idx]
            
            # Create error analysis DataFrame
            error_df = X_test.copy()
            error_df['Actual'] = y_test
            error_df['Predicted'] = y_pred
            error_df['Error'] = y_test != y_pred
            
            # Filter errors
            errors_only = error_df[error_df['Error']]
            
            # Display errors
            st.markdown(f"#### Errors ({len(errors_only)} out of {len(X_test)} samples)")
            
            # Categorize errors
            false_positives = errors_only[errors_only['Predicted'] == 1]
            false_negatives = errors_only[errors_only['Predicted'] == 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**False Positives: {len(false_positives)}**")
                st.write("Companies incorrectly predicted as bankrupt")
                if not false_positives.empty:
                    st.dataframe(false_positives.head(10))
            
            with col2:
                st.markdown(f"**False Negatives: {len(false_negatives)}**")
                st.write("Bankrupt companies missed by the model")
                if not false_negatives.empty:
                    st.dataframe(false_negatives.head(10))
            
            # Error distribution by feature
            st.markdown("#### Error Distribution by Feature")
            
            # Select feature for error analysis
            feature_names = X_test.columns
            selected_feature = st.selectbox("Select feature", feature_names)
            
            # Plot feature distribution for errors vs correct predictions
            fig, ax = plt.subplots(figsize=(10, 6))
            
            correct = error_df[~error_df['Error']][selected_feature]
            incorrect = error_df[error_df['Error']][selected_feature]
            
            ax.hist([correct, incorrect], bins=20, alpha=0.7, 
                   label=['Correct Predictions', 'Errors'])
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {selected_feature} for Correct vs Incorrect Predictions')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            st.markdown("### Sample Predictions")
            
            # Create prediction comparison DataFrame for all models
            sample_df = X_test.sample(min(10, len(X_test))).copy()
            sample_df['Actual'] = y_test.iloc[sample_df.index].values
            
            for i, model_name in enumerate(get_model_names()):
                sample_df[f'{model_name} Pred'] = all_predictions[i][sample_df.index]
                sample_df[f'{model_name} Prob'] = all_probabilities[i][sample_df.index]
            
            # Display sample predictions
            st.dataframe(sample_df)
            
            # Let user try custom inputs
            st.markdown("#### Try Your Own Input")
            
            # Create input fields for each feature
            input_data = {}
            col1, col2, col3 = st.columns(3)
            
            feature_names = X_test.columns
            
            for i, feature in enumerate(feature_names):
                # Determine column
                if i % 3 == 0:
                    col = col1
                elif i % 3 == 1:
                    col = col2
                else:
                    col = col3
                
                # Calculate min, max, and mean values
                min_val = X_test[feature].min()
                max_val = X_test[feature].max()
                mean_val = X_test[feature].mean()
                
                # Create input field
                input_data[feature] = col.number_input(
                    feature,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(mean_val),
                    format="%.2f"
                )
            
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            # Make predictions
            if st.button("Predict"):
                # Display predictions for all models
                pred_results = {}
                
                for i, model_name in enumerate(get_model_names()):
                    model = models[i]
                    
                    # Special handling for pipeline models
                    if model_name in ["Logistic Regression", "SVM", "KNN"]:
                        pred = model.predict(input_df)[0]
                        prob = model.predict_proba(input_df)[0][1]
                    else:
                        pred = model.predict(input_df)[0]
                        prob = model.predict_proba(input_df)[0][1]
                    
                    pred_results[model_name] = {
                        "Prediction": "Bankrupt" if pred == 1 else "Alive",
                        "Bankruptcy Probability": f"{prob:.4f}"
                    }
                
                # Display results
                results_df = pd.DataFrame(pred_results).T
                st.dataframe(results_df)
    else:
        st.warning("No results available. Please load data and train models.")

if __name__ == "__main__":
    main()
