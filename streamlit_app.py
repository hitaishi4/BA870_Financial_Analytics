import streamlit as st
from tabs import (
    load_data,
    split_data,
    train_models,
    get_metrics,
    get_feature_importances,
    get_permutation_importances,
    show_overview_tab,
    show_training_tab,
    show_comparison_tab,
    show_roc_tab,
    show_confusion_tab,
    show_classification_tab,
    show_feature_importance_tab,
    show_permutation_tab
)

def main():
    st.title("📊 American Bankruptcy Analysis")

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, target_col="Bankruptcy")

    models = train_models(X_train, y_train)
    metrics = get_metrics(models, X_test, y_test)

    feat_imps = get_feature_importances(models, feature_names=X_train.columns.tolist())
    perm_imps = get_permutation_importances(models, X_test, y_test, feature_names=X_train.columns.tolist())

    tabs = st.tabs([
        "Overview",
        "Training Results",
        "AUC Comparison",
        "ROC Curves",
        "Confusion Matrices",
        "Classification Reports",
        "Feature Importances",
        "Permutation Importances"
    ])

    with tabs[0]:
        show_overview_tab(df)
    with tabs[1]:
        show_training_tab(metrics)
    with tabs[2]:
        show_comparison_tab(metrics)
    with tabs[3]:
        show_roc_tab(metrics)
    with tabs[4]:
        show_confusion_tab(metrics)
    with tabs[5]:
        show_classification_tab(metrics)
    with tabs[6]:
        show_feature_importance_tab(feat_imps)
    with tabs[7]:
        show_permutation_tab(perm_imps)

if __name__ == "__main__":
    main()
