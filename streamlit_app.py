import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Basic page setup
st.set_page_config(page_title="Bankruptcy Dashboard", layout="wide")
st.title("Bankruptcy Prediction Dashboard")

# Try to load data
try:
    # Define paths to try
    paths = ['data/american_bankruptcy.csv', 'american_bankruptcy.csv']
    data_loaded = False
    
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.success(f"Data loaded successfully from {path}")
            st.write(f"Number of records: {len(df)}")
            st.write(f"Columns: {', '.join(df.columns.tolist())}")
            data_loaded = True
            break
    
    if not data_loaded:
        st.error("Could not find data file. Please check file path.")
        
except Exception as e:
    st.error(f"Error: {e}")
