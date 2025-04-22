import streamlit as st
from tabs import (
    load_data, split_data,
    train_models, get_metrics,
    show_data_tab, show_training_tab,
    show_comparison_tab, show_roc_tab,
    show_confusion_tab, show_classification_tab,
    show_prediction_tab
)

def main():
    st.title("📊 American Bankruptcy Analysis")

    # Load and preprocess data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Dataset not found. Place 'american_bankruptcy.csv' under 'data/' folder.")
        return

    target = "Bankruptcy"
    X_train, X_test, y_train, y_test = split_data(df, target)
    models = train_models(X_train, y_train)
    if 'results' not in st.session_state:
        st.session_state['results'] = get_metrics(models, X_test, y_test)

    tabs = st.tabs([
        "Data", "Training Results", "AUC Comparison",
        "ROC Curves", "Confusion Matrices",
        "Classification Reports", "Prediction"
    ])

    with tabs[0]:
        show_data_tab(df)
    with tabs[1]:
        show_training_tab(st.session_state['results'])
    with tabs[2]:
        show_comparison_tab(st.session_state['results'])
    with tabs[3]:
        show_roc_tab(st.session_state['results'])
    with tabs[4]:
        show_confusion_tab(st.session_state['results'])
    with tabs[5]:
        show_classification_tab(st.session_state['results'])
    with tabs[6]:
        features = df.drop(columns=[target]).columns.tolist()
        show_prediction_tab(models, features)

if __name__ == "__main__":
    main()
