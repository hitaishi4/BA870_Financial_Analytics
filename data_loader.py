import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_data(file_path):
    """
    Load the bankruptcy dataset from the specified file path.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the bankruptcy data
    
    Returns:
    --------
    pd.DataFrame
        The loaded bankruptcy dataset
    """
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
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def preprocess_data(df):
    """
    Preprocess the bankruptcy dataset for model training and evaluation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The bankruptcy dataset
    
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, feature_names
    """
    try:
        # Ensure the Bankruptcy column exists
        if 'Bankruptcy' not in df.columns:
            if 'status_label' in df.columns:
                df['Bankruptcy'] = df['status_label'].map({'failed': 1, 'alive': 0})
            else:
                raise ValueError("Missing both 'Bankruptcy' and 'status_label' columns")
        
        # Get feature names - these are the financial indicators
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
        
        # If columns are named X1, X2, etc., rename them
        if "X1" in df.columns:
            df = df.rename(columns=rename_map)
        
        features = list(rename_map.values())
        
        # Make sure all expected features are in the dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing expected features: {missing_features}")
        
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
        raise Exception(f"Error preprocessing data: {str(e)}")

def generate_sample_data(n_samples=1000, bankruptcy_rate=0.05):
    """
    Generate synthetic bankruptcy data for testing or demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    bankruptcy_rate : float
        Proportion of bankrupt companies in the dataset
    
    Returns:
    --------
    pd.DataFrame
        Synthetic bankruptcy dataset
    """
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

def load_or_generate_data(file_path=None):
    """
    Load data from a file or generate synthetic data if the file doesn't exist.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the data file (if None, generates synthetic data)
    
    Returns:
    --------
    pd.DataFrame
        The bankruptcy dataset
    """
    if file_path is not None and os.path.exists(file_path):
        try:
            return load_data(file_path)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            print("Falling back to synthetic data generation")
    
    # Generate synthetic data
    return generate_sample_data()
