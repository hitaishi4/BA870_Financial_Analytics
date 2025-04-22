import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
import io
import base64

def plot_roc_curves(all_probabilities, y_test, model_names):
    """
    Plot ROC curves for multiple models.
    
    Parameters:
    -----------
    all_probabilities : list
        List of probability predictions from each model
    y_test : pd.Series or np.array
        True labels
    model_names : list
        Names of the models
    
    Returns:
    --------
    matplotlib.figure.Figure
        The ROC curve plot
    """
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
    
    # Add AUC ranking in the corner
    auc_ranking = pd.Series(auc_values, index=model_names).sort_values(ascending=False)
    best_model = auc_ranking.index[0]
    worst_model = auc_ranking.index[-1]
    
    # Add text annotation
    ax.text(
        0.05, 0.05, 
        f"Best: {best_model} (AUC = {auc_ranking.iloc[0]:.3f})\nWorst: {worst_model} (AUC = {auc_ranking.iloc[-1]:.3f})",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    return fig

def plot_feature_importances(importances, title=None, n_features=None):
    """
    Plot feature importances.
    
    Parameters:
    -----------
    importances : pd.Series
        Feature importance values indexed by feature names
    title : str, optional
        Plot title
    n_features : int, optional
        Number of top features to display (default: all)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The feature importance plot
    """
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
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    title : str, optional
        Plot title
    
    Returns:
    --------
    matplotlib.figure.Figure
        The confusion matrix plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add colorbar
    ax.figure.colorbar(im, ax=ax)
    
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
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Annotate cells
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    # Display accuracy
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.text(-0.1, -0.1, f"Accuracy: {accuracy:.3f}",
            transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    return fig
