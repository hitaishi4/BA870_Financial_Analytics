import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import streamlit as st
import io
import base64
import os
import time

def get_model_names():
    """
    Get list of model names used in the dashboard.
    
    Returns:
    --------
    list
        List of model names
    """
    return [
        "Decision Tree",
        "Gradient Boosting",
        "Random Forest",
        "Logistic Regression",
        "SVM",
        "KNN"
    ]

def get_metrics_df(results):
    """
    Create a DataFrame of performance metrics for all models.
    
    Parameters:
    -----------
    results : dict
        Dictionary of model results
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with models as index and metrics as columns
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    metrics_df = pd.DataFrame(index=results.keys(), columns=metrics)
    
    for model_name, model_results in results.items():
        metrics_df.loc[model_name, 'Accuracy'] = model_results.get('accuracy', np.nan)
        metrics_df.loc[model_name, 'Precision'] = model_results.get('precision', np.nan)
        metrics_df.loc[model_name, 'Recall'] = model_results.get('recall', np.nan)
        metrics_df.loc[model_name, 'F1 Score'] = model_results.get('f1', np.nan)
        metrics_df.loc[model_name, 'AUC'] = model_results.get('auc', np.nan)
    
    return metrics_df

def get_best_model(metrics_df, metric='AUC'):
    """
    Get the best performing model based on a specific metric.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with models as index and metrics as columns
    metric : str
        Metric to use for comparison
    
    Returns:
    --------
    str
        Name of the best performing model
    """
    return metrics_df[metric].idxmax()

def format_confusion_matrix(cm):
    """
    Format confusion matrix for display.
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    
    Returns:
    --------
    pd.DataFrame
        Formatted confusion matrix DataFrame
    """
    return pd.DataFrame(
        cm,
        index=['Actual Alive', 'Actual Bankrupt'],
        columns=['Predicted Alive', 'Predicted Bankrupt']
    )

def calculate_metrics_at_threshold(y_true, y_proba, threshold=0.5):
    """
    Calculate performance metrics at a specific probability threshold.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Probability predictions
    threshold : float
        Probability threshold
    
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def find_optimal_threshold(y_true, y_proba, criterion='f1'):
    """
    Find optimal probability threshold based on a specific criterion.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Probability predictions
    criterion : str
        Criterion to optimize ('f1', 'precision', 'recall', 'accuracy')
    
    Returns:
    --------
    float
        Optimal threshold
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    if criterion == 'f1':
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * precision * recall / (precision + recall)
        optimal_idx = int(np.nanargmax(f1_scores))
        return float(thresholds_pr[optimal_idx]) if optimal_idx < len(thresholds_pr) else 0.5
    metrics = []
    for thresh in thresholds:
        metrics.append(calculate_metrics_at_threshold(y_true, y_proba, thresh))
    df = pd.DataFrame(metrics)
    return float(df.loc[df[criterion].idxmax(), 'threshold'])

def get_feature_ranking_comparison(models, feature_names):
    """
    Compare feature importance rankings across models.
    
    Parameters:
    -----------
    models : list
        List of trained model objects
    feature_names : list
        Feature names
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature rankings for each model
    """
    rankings = {}
    # Decision Tree
    if hasattr(models[0], 'feature_importances_'):
        rankings['Decision Tree'] = pd.Series(models[0].feature_importances_, index=feature_names).rank(ascending=False).astype(int)
    # Gradient Boosting
    if hasattr(models[1], 'feature_importances_'):
        rankings['Gradient Boosting'] = pd.Series(models[1].feature_importances_, index=feature_names).rank(ascending=False).astype(int)
    # Random Forest
    if hasattr(models[2], 'feature_importances_'):
        rankings['Random Forest'] = pd.Series(models[2].feature_importances_, index=feature_names).rank(ascending=False).astype(int)
    # Logistic Regression
    if hasattr(models[3], 'named_steps') and hasattr(models[3].named_steps.get('logreg', None), 'coef_'):
        rankings['Logistic Regression'] = pd.Series(np.abs(models[3].named_steps['logreg'].coef_[0]), index=feature_names).rank(ascending=False).astype(int)
    df = pd.DataFrame(rankings)
    if not df.empty:
        df['Average Rank'] = df.mean(axis=1)
        df = df.sort_values('Average Rank')
    return df

def get_prediction_probabilities(models, X):
    """
    Get bankruptcy probability predictions from all models.
    
    Parameters:
    -----------
    models : list
        List of trained model objects
    X : pd.DataFrame
        Feature data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with probability predictions for each model
    """
    df = pd.DataFrame(index=X.index)
    for model, name in zip(models, get_model_names()):
        df[name] = model.predict_proba(X)[:, 1]
    return df

def calculate_ensemble_predictions(proba_df, weights=None, threshold=0.5):
    """
    Calculate ensemble predictions based on weighted average of probabilities.
    
    Parameters:
    -----------
    proba_df : pd.DataFrame
        DataFrame with probability predictions for each model
    weights : dict or None
        Dictionary mapping model names to weights
    threshold : float
        Probability threshold for classification
    
    Returns:
    --------
    tuple
        (ensemble_proba, ensemble_pred)
    """
    if weights is None:
        weights = {m:1.0 for m in proba_df.columns}
    total = sum(weights.values())
    weights = {m: w/total for m,w in weights.items()}
    ensemble_proba = sum(proba_df[m]*weights[m] for m in proba_df.columns)
    ensemble_pred = (ensemble_proba>=threshold).astype(int)
    return ensemble_proba, ensemble_pred

def timeit(func):
    """
    Decorator to measure function execution time.
    
    Parameters:
    -----------
    func : function
        Function to measure
    
    Returns:
    --------
    function
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.2f}s")
        return result
    return wrapper

def create_download_link(df, filename="data.csv", text="Download data"):
    """
    Create a download link for a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to download
    filename : str
        Download filename
    text : str
        Link text
    
    Returns:
    --------
    str
        HTML download link
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

def show_code(file_path):
    """
    Display code from a file with syntax highlighting.
    
    Parameters:
    -----------
    file_path : str
        Path to the code file
    """
    try:
        with open(file_path,'r') as f:
            code = f.read()
        ext = os.path.splitext(file_path)[1]
        lang = {'.py':'python'}.get(ext,'text')
        st.code(code, language=lang)
    except Exception as e:
        st.error(f"Error displaying code: {e}")

def get_class_weights(y):
    """
    Calculate class weights inversely proportional to class frequencies.
    
    Parameters:
    -----------
    y : array-like
        Target labels
    
    Returns:
    --------
    dict
        Class weights
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {c: total/(len(classes)*cnt) for c,cnt in zip(classes,counts)}

def get_top_features(feature_importances, n=5):
    """
    Get top n most important features.
    
    Parameters:
    -----------
    feature_importances : pd.Series
        Feature importance values indexed by feature names
    n : int
        Number of top features to return
    
    Returns:
    --------
    list
        Names of top n features
    """
    return feature_importances.sort_values(ascending=False).head(n).index.tolist()

def format_financial_ratio(value):
    """
    Format financial ratio for display.
    
    Parameters:
    -----------
    value : float
        Financial ratio value
    
    Returns:
    --------
    str
        Formatted ratio
    """
    if value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"

def calculate_risk_score(probabilities, thresholds=None):
    """
    Calculate risk score from bankruptcy probabilities.
    
    Parameters:
    -----------
    probabilities : array-like
        Bankruptcy probability predictions
    thresholds : list, optional
        Risk score thresholds
    
    Returns:
    --------
    array-like
        Risk scores (1-10)
    """
    if thresholds is None:
        thresholds = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    scores = np.ones(len(probabilities),dtype=int)
    for i,thr in enumerate(thresholds):
        scores[probabilities>=thr] = i+2
    return scores

def highlight_threshold(df, threshold=0.5, column='Probability'):
    """
    Apply styling to highlight values above threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to style
    threshold : float
        Threshold value
    column : str
        Column to highlight
    
    Returns:
    --------
    pd.DataFrame.style
        Styled DataFrame
    """
    mask = df[column]>=threshold
    return df.style.applymap(lambda v: 'color:red' if v>=threshold else 'color:green', subset=[column])

def create_cached_func(func):
    """
    Create a cached version of a function using Streamlit's caching.
    
    Parameters:
    -----------
    func : function
        Function to cache
    
    Returns:
    --------
    function
        Cached function
    """
    return st.cache_data(func)

def format_time_period(start_year, end_year):
    """
    Format time period for display.
    
    Parameters:
    -----------
    start_year : int
        Start year
    end_year : int
        End year
    
    Returns:
    --------
    str
        Formatted time period
    """
    return f"{start_year}-{end_year}"
