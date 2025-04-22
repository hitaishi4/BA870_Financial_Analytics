import numpy as np
import pandas as pd
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

def train_models(X_train, y_train):
    """
    Train multiple bankruptcy prediction models.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature data
    y_train : pd.Series
        Training target data (bankruptcy indicators)
    
    Returns:
    --------
    list
        List of trained model objects
    """
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
    """
    Evaluate multiple bankruptcy prediction models.
    
    Parameters:
    -----------
    models : list
        List of trained model objects
    X_test : pd.DataFrame
        Testing feature data
    y_test : pd.Series
        Testing target data (bankruptcy indicators)
    feature_names : list, optional
        Names of features for feature importance analysis
        
    Returns:
    --------
    tuple
        (results_dict, all_predictions, all_probabilities)
    """
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

def get_feature_importance(model, feature_names, model_type=None):
    """
    Extract feature importance from a trained model.
    
    Parameters:
    -----------
    model : object
        Trained model object
    feature_names : list
        List of feature names
    model_type : str, optional
        Type of model ('dt', 'gb', 'rf', 'lr')
    
    Returns:
    --------
    pd.Series
        Feature importance values indexed by feature names
    """
    if model_type is None:
        # Try to infer model type
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('logreg', None), 'coef_'):
            importances = np.abs(model.named_steps['logreg'].coef_[0])
        else:
            raise ValueError("Unknown model type, cannot extract feature importances")
    else:
        # Extract based on provided model type
        if model_type in ['dt', 'gb', 'rf']:
            importances = model.feature_importances_
        elif model_type == 'lr':
            if hasattr(model, 'named_steps'):
                importances = np.abs(model.named_steps['logreg'].coef_[0])
            else:
                importances = np.abs(model.coef_[0])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)

def evaluate_feature_subsets(X_train, y_train, X_test, y_test, feature_importances, n_features_list=None):
    """
    Evaluate model performance with different subsets of top features.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature data
    y_train : pd.Series
        Training target data
    X_test : pd.DataFrame
        Testing feature data
    y_test : pd.Series
        Testing target data
    feature_importances : pd.Series
        Feature importance scores
    n_features_list : list, optional
        List of number of top features to evaluate (default: [5, 10, 15, 'all'])
    
    Returns:
    --------
    pd.DataFrame
        Performance metrics for different feature subsets
    """
    if n_features_list is None:
        n_features_list = [5, 10, 15, 'all']
    
    results = []
    
    # Get ordered feature names
    ordered_features = feature_importances.index.tolist()
    
    for n_features in n_features_list:
        if n_features == 'all':
            # Use all features
            selected_features = ordered_features
        else:
            # Use top n features
            selected_features = ordered_features[:n_features]
        
        # Train a gradient boosting model on selected features
        gb_clf = GradientBoostingClassifier(random_state=42)
        gb_clf.fit(X_train[selected_features], y_train)
        
        # Evaluate
        y_pred = gb_clf.predict(X_test[selected_features])
        y_proba = gb_clf.predict_proba(X_test[selected_features])[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Store results
        results.append({
            'n_features': 'All' if n_features == 'all' else n_features,
            'features': selected_features,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc
        })
    
    return pd.DataFrame(results)

def get_model_threshold_performance(model, X_test, y_test, thresholds=None):
    """
    Evaluate model performance at different probability thresholds.
    
    Parameters:
    -----------
    model : object
        Trained model object
    X_test : pd.DataFrame
        Testing feature data
    y_test : pd.Series
        Testing target data
    thresholds : list, optional
        List of probability thresholds to evaluate
    
    Returns:
    --------
    pd.DataFrame
        Performance metrics at different thresholds
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    
    for threshold in thresholds:
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        })
    
    return pd.DataFrame(results)
