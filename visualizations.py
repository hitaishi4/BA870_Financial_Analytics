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
    
    # Add precision, recall in the title (if not provided)
    if title is None:
        # Calculate precision and recall
        precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
        recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        
        ax.set_title(
            f'Confusion Matrix\nPrecision: {precision:.3f}, Recall: {recall:.3f}',
            fontsize=16
        )
    
    # Calculate and display overall metrics
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.text(
        -0.1, -0.1, 
        f"Accuracy: {accuracy:.3f}",
        transform=ax.transAxes,
        fontsize=12
    )
    
    plt.tight_layout()
    return fig

def plot_model_comparison(metrics_df, selected_metrics):
    """
    Plot comparison of model performance for selected metrics.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with models as index and metrics as columns
    selected_metrics : list
        List of metric names to plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The model comparison plot
    """
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

def plot_feature_correlation_matrix(X, feature_names=None):
    """
    Plot correlation matrix of features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature data
    feature_names : list, optional
        Feature names to include (default: all)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The correlation matrix plot
    """
    # Convert to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)
    
    # Filter features if specified
    if feature_names is not None:
        X = X[feature_names]
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    ax.set_yticklabels(corr_matrix.columns)
    
    # Loop over data dimensions and create text annotations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(
                j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                ha="center", va="center",
                color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                fontsize=8
            )
    
    ax.set_title('Feature Correlation Matrix', fontsize=16)
    
    plt.tight_layout()
    return fig

def plot_class_distribution(y, title=None):
    """
    Plot class distribution (bankruptcy vs non-bankruptcy).
    
    Parameters:
    -----------
    y : array-like
        Target labels
    title : str, optional
        Plot title
    
    Returns:
    --------
    matplotlib.figure.Figure
        The class distribution plot
    """
    # Count classes
    classes, counts = np.unique(y, return_counts=True)
    class_names = ['Alive', 'Bankrupt']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create bar chart
    bars = ax.bar(
        [class_names[c] for c in classes],
        counts,
        color=['#395c40', '#a63603']
    )
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.1,
            f'{height}',
            ha='center',
            va='bottom'
        )
    
    # Add percentage labels
    total = len(y)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = height / total * 100
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height / 2,
            f'{percentage:.1f}%',
            ha='center',
            va='center',
            color='white',
            fontweight='bold'
        )
    
    # Set plot attributes
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title if title else 'Class Distribution', fontsize=16)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_feature_distribution(X, feature, y=None):
    """
    Plot distribution of a feature, optionally split by class.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature data
    feature : str
        Feature name to plot
    y : array-like, optional
        Target labels
    
    Returns:
    --------
    matplotlib.figure.Figure
        The feature distribution plot
    """
    # Convert to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if y is not None:
        # Split by class
        classes = np.unique(y)
        class_names = ['Alive', 'Bankrupt']
        colors = ['#395c40', '#a63603']
        
        # Create histogram for each class
        for c, color, name in zip(classes, colors, class_names):
            ax.hist(
                X.loc[y == c, feature],
                bins=20,
                alpha=0.7,
                color=color,
                label=name
            )
            
            # Add mean line
            mean = X.loc[y == c, feature].mean()
            ax.axvline(
                mean,
                color=color,
                linestyle='--',
                alpha=0.8,
                label=f'{name} Mean'
            )
        
        ax.legend()
    else:
        # Single histogram
        ax.hist(
            X[feature],
            bins=20,
            alpha=0.7,
            color='#395c40'
        )
        
        # Add mean line
        mean = X[feature].mean()
        ax.axvline(
            mean,
            color='red',
            linestyle='--',
            alpha=0.8,
            label=f'Mean'
        )
        
        ax.legend()
    
    # Set plot attributes
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of {feature}', fontsize=16)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_permutation_importance(model, X, y, feature_names, n_repeats=10):
    """
    Plot feature importance using permutation importance method.
    
    Parameters:
    -----------
    model : object
        Trained model
    X : pd.DataFrame
        Feature data
    y : array-like
        Target labels
    feature_names : list
        Feature names
    n_repeats : int
        Number of times to permute each feature
    
    Returns:
    --------
    matplotlib.figure.Figure
        The permutation importance plot
    """
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42
    )
    
    # Create Series with feature names
    importances = pd.Series(
        perm_importance.importances_mean,
        index=feature_names
    ).sort_values(ascending=False)
    
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
            width + 0.001,
            bar.get_y() + bar.get_height()/2,
            f'{width:.4f}',
            ha='left',
            va='center'
        )
    
    # Set plot attributes
    ax.set_xlabel('Permutation Importance', fontsize=12)
    ax.set_title('Feature Importance (Permutation Method)', fontsize=16)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_threshold_performance(thresholds, metrics):
    """
    Plot model performance at different probability thresholds.
    
    Parameters:
    -----------
    thresholds : array-like
        Probability thresholds
    metrics : dict
        Dict of metric values at each threshold
    
    Returns:
    --------
    matplotlib.figure.Figure
        The threshold performance plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each metric
    for metric_name, values in metrics.items():
        ax.plot(
            thresholds,
            values,
            marker='o',
            label=metric_name,
            linewidth=2
        )
    
    # Set plot attributes
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Model Performance vs. Classification Threshold', fontsize=16)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Add vertical line at default threshold (0.5)
    ax.axvline(
        x=0.5,
        color='black',
        linestyle='--',
        alpha=0.5,
        label='Default Threshold'
    )
    
    plt.tight_layout()
    return fig

def plot_model_comparison_radar(metrics_df, selected_models=None):
    """
    Create a radar chart to compare different models across metrics.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with models as index and metrics as columns
    selected_models : list, optional
        List of model names to include (default: all)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The radar chart
    """
    # Filter models if specified
    if selected_models is not None:
        metrics_df = metrics_df.loc[selected_models]
    
    # Get metrics and models
    metrics = metrics_df.columns.tolist()
    models = metrics_df.index.tolist()
    
    # Number of metrics
    N = len(metrics)
    
    # Create angle array (divide the plot / number of metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    # Make the plot circular
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Add each model
    for i, model in enumerate(models):
        # Get metric values for model and make circular
        values = metrics_df.loc[model].tolist()
        values += values[:1]
        
        # Plot model
        ax.plot(
            angles,
            values,
            linewidth=2,
            label=model
        )
        
        # Fill area
        ax.fill(
            angles,
            values,
            alpha=0.1
        )
    
    # Set labels for each metric
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    return fig
