import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time

# Import custom modules
from data_loader import load_data, preprocess_data
from model_trainer import train_models, evaluate_models
from visualizations import (
    plot_roc_curves, plot_feature_importances, 
    plot_confusion_matrices, plot_model_comparison
)
from utils import get_model_names, get_metrics_df

# Page configuration
st.set_page_config(
    page_title="Bankruptcy Prediction Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Add custom CSS
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
            # Use the existing data loading function
            try:
                df = load_data("american_bankruptcy.csv")
                st.sidebar.success("Sample data loaded!")
            except Exception as e:
                st.sidebar.error(f"Error loading sample data: {e}")
                st.sidebar.info("Using backup sample data instead")
                # Generate backup sample data if file not available
                df = generate_sample_data()
    
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
        st.button("Explore Data", on_click=lambda: st.session_state.update({"page": "Data Exploration"}))
    with col2:
        st.button("Compare Models", on_click=lambda: st.session_state.update({"page": "Model Comparison"}))
    with col3:
        st.button("Analyze Features", on_click=lambda: st.session_state.update({"page": "Feature Importance"}))

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

def generate_sample_data():
    """Generate backup sample data if file is not available"""
    np.random.seed(42)
    
    # Generate 1000 samples with 18 features
    n_samples = 1000
    n_features = 18
    
    # Create sample data
    X = np.random.rand(n_samples, n_features) * 10
    
    # Feature names from the original code
    feature_names = [
        "Current Assets", "Cost of Goods Sold", "D&A", "EBITDA",
        "Inventory", "Net Income", "Total Receivables", "Market Value",
        "Net Sales", "Total Assets", "Total Long-term Debt", "EBIT",
        "Gross Profit", "Total Current Liabilities", "Retained Earnings",
        "Total Revenue", "Total Liabilities", "Total Operating Expenses"
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add year and status columns
    years = np.random.choice(range(1999, 2019), n_samples)
    df['year'] = years
    
    # Generate bankruptcy status with 5% bankruptcy rate
    bankruptcy = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    df['Bankruptcy'] = bankruptcy
    
    # Add status_label column
    df['status_label'] = ['failed' if b == 1 else 'alive' for b in bankruptcy]
    
    return df

if __name__ == "__main__":
    main()
