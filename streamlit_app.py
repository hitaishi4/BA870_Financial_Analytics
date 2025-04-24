import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from io import StringIO
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Bankruptcy Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
PRIMARY_COLOR = "#395c40"
SECONDARY_COLOR = "#98ba66"
ACCENT_COLOR = "#ff4c4b"
DATA_PATHS = [
    'data/american_bankruptcy.csv',
    'american_bankruptcy.csv',
    '../data/american_bankruptcy.csv',
    './american_bankruptcy.csv',
]

# Custom CSS with improved accessibility
st.markdown(f"""
<style>
:root {{
    --primary: {PRIMARY_COLOR};
    --secondary: {SECONDARY_COLOR};
    --accent: {ACCENT_COLOR};
}}

.main-header {{
    font-size: 2.5rem !important;
    color: var(--primary);
    text-align: center;
    margin: 1rem 0;
}}

.section-header {{
    font-size: 1.8rem !important;
    color: var(--primary);
    margin: 1.5rem 0 1rem;
}}

.metric-card {{
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

.data-table {{
    margin: 1rem 0;
    overflow-x: auto;
}}

.alert {{
    padding: 1rem;
    border-radius: 4px;
    margin: 1rem 0;
}}

.alert-warning {{
    background: #fff3cd;
    color: #856404;
}}

.alert-error {{
    background: #f8d7da;
    color: #721c24;
}}

/* Improved accessibility for color contrast */
[data-testid="stDataFrame"] {{
    color: #333 !important;
}}

/* Animation enhancements */
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.stApp > div {{
    animation: fadeIn 0.5s ease-out;
}}
</style>
""", unsafe_allow_html=True)

# Data loading with enhanced error handling
@st.cache_data(show_spinner="Loading financial data...")
def load_financial_data():
    """Load and validate financial data with comprehensive error handling"""
    error_log = StringIO()
    df = pd.DataFrame()
    
    for path in DATA_PATHS:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                logger.info(f"Successfully loaded data from {path}")
                
                # Column normalization
                col_mapping = {
                    "X1": "Current Assets", "X2": "Cost of Goods Sold",
                    "X3": "D&A", "X4": "EBITDA", "X5": "Inventory",
                    "X6": "Net Income", "X7": "Total Receivables",
                    "X8": "Market Value", "X9": "Net Sales",
                    "X10": "Total Assets", "X11": "Total Long-term Debt",
                    "X12": "EBIT", "X13": "Gross Profit",
                    "X14": "Total Current Liabilities",
                    "X15": "Retained Earnings", "X16": "Total Revenue",
                    "X17": "Total Liabilities", "X18": "Total Operating Expenses"
                }
                
                df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
                
                # Handle target variable
                if 'status_label' in df.columns:
                    df['Bankrupt'] = df['status_label'].apply(
                        lambda x: 1 if x.lower() in ['failed', 'bankrupt'] else 0
                    )
                elif 'Bankruptcy' in df.columns:
                    df.rename(columns={'Bankruptcy': 'Bankrupt'}, inplace=True)
                
                # Validate critical columns
                required_cols = ['Current Assets', 'Total Current Liabilities',
                                'Retained Earnings', 'Total Assets', 'EBIT',
                                'Market Value', 'Total Liabilities', 'Net Sales']
                missing = [c for c in required_cols if c not in df.columns]
                
                if missing:
                    raise ValueError(f"Missing required columns: {missing}")
                
                return df
        except Exception as e:
            error_log.write(f"Error loading {path}: {str(e)}\n")
    
    st.error("Failed to load data. Please check file paths and data format.")
    logger.error(f"Data loading errors:\n{error_log.getvalue()}")
    return pd.DataFrame()

# Enhanced Z-Score calculator with dynamic thresholds
class ZScoreAnalyzer:
    def __init__(self, data, distress_thresh=1.8, safe_thresh=2.99):
        self.data = data
        self.distress_thresh = distress_thresh
        self.safe_thresh = safe_thresh
        self.required_cols = [
            'Current Assets', 'Total Current Liabilities',
            'Retained Earnings', 'Total Assets', 'EBIT',
            'Market Value', 'Total Liabilities', 'Net Sales'
        ]
    
    def calculate(self):
        try:
            df = self.data[self.required_cols].copy()
            df['T1'] = (df['Current Assets'] - df['Total Current Liabilities']) / df['Total Assets']
            df['T2'] = df['Retained Earnings'] / df['Total Assets']
            df['T3'] = df['EBIT'] / df['Total Assets']
            df['T4'] = df['Market Value'] / df['Total Liabilities']
            df['T5'] = df['Net Sales'] / df['Total Assets']
            
            df['Z-Score'] = 1.2*df['T1'] + 1.4*df['T2'] + 3.3*df['T3'] + 0.6*df['T4'] + 0.99*df['T5']
            
            df['Status'] = pd.cut(
                df['Z-Score'],
                bins=[-np.inf, self.distress_thresh, self.safe_thresh, np.inf],
                labels=['Distress', 'Grey', 'Safe'],
                right=False
            )
            
            return df
        except KeyError as e:
            st.error(f"Missing required column: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error calculating Z-Scores: {str(e)}")
            return pd.DataFrame()

# Model evaluation metrics
MODEL_METRICS = {
    'Decision Tree': {'accuracy': 0.8925, 'precision': 0.0589, 'recall': 0.2404, 'f1': 0.0947, 'auc': 0.574},
    'Gradient Boosting': {'accuracy': 0.9761, 'precision': 0.3846, 'recall': 0.0348, 'f1': 0.0639, 'auc': 0.827},
    'Random Forest': {'accuracy': 0.9759, 'precision': 0.3200, 'recall': 0.0279, 'f1': 0.0513, 'auc': 0.835},
    'Logistic Regression': {'accuracy': 0.9752, 'precision': 0.3125, 'recall': 0.0523, 'f1': 0.0896, 'auc': 0.827},
    'SVM': {'accuracy': 0.9765, 'precision': 0.3333, 'recall': 0.0070, 'f1': 0.0137, 'auc': 0.590},
    'KNN': {'accuracy': 0.9589, 'precision': 0.1414, 'recall': 0.1498, 'f1': 0.1455, 'auc': 0.695}
}

# Initialize session state
def init_session():
    if 'data' not in st.session_state:
        st.session_state.data = load_financial_data()
    if 'zscore' not in st.session_state:
        st.session_state.zscore = None

# Main app structure
def main():
    init_session()
    
    st.markdown('<h1 class="main-header">Corporate Financial Health Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select Analysis", [
            "Data Overview", 
            "Model Performance",
            "Financial Health Assessment",
            "Comparative Analysis"
        ])
        
        st.markdown("---")
        st.header("Settings")
        distress_thresh = st.slider("Distress Threshold", 1.0, 2.5, 1.8)
        safe_thresh = st.slider("Safe Threshold", 2.0, 4.0, 2.99)
        
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.session_state.data = load_financial_data()
            st.rerun()
    
    # Page routing
    if page == "Data Overview":
        show_data_overview()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Financial Health Assessment":
        show_financial_health(distress_thresh, safe_thresh)
    elif page == "Comparative Analysis":
        show_comparative_analysis()

# Page components
def show_data_overview():
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    if st.session_state.data.empty:
        st.warning("No data loaded. Check data source or file format.")
        return
    
    with st.expander("Dataset Summary", expanded=True):
        cols = st.columns([1,2])
        cols[0].metric("Total Companies", len(st.session_state.data))
        cols[1].metric("Data Columns", len(st.session_state.data.columns))
        
        if 'Bankrupt' in st.session_state.data:
            bankrupt_pct = st.session_state.data['Bankrupt'].mean() * 100
            st.progress(bankrupt_pct/100, text=f"Bankruptcy Rate: {bankrupt_pct:.1f}%")
    
    with st.expander("Data Preview"):
        st.dataframe(st.session_state.data.sample(5, random_state=42))
    
    with st.expander("Financial Health Distribution"):
        if 'Bankrupt' in st.session_state.data:
            fig, ax = plt.subplots(figsize=(8,4))
            st.session_state.data['Bankrupt'].value_counts().plot(
                kind='pie', 
                ax=ax, 
                labels=['Healthy', 'Distressed'],
                colors=[SECONDARY_COLOR, ACCENT_COLOR],
                autopct='%1.1f%%'
            )
            plt.ylabel('')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.error("Bankruptcy status data not found")

def show_model_performance():
    st.markdown('<h2 class="section-header">Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown("### Comparative Metrics")
        metrics_df = pd.DataFrame(MODEL_METRICS).T
        st.dataframe(
            metrics_df.style.format("{:.3f}").highlight_max(color=SECONDARY_COLOR),
            use_container_width=True
        )
    
    with col2:
        st.markdown("### Performance Visualization")
        metric = st.selectbox("Select Metric", ['accuracy', 'precision', 'recall', 'f1', 'auc'])
        
        fig, ax = plt.subplots(figsize=(8,4))
        metrics_df[metric].sort_values().plot(
            kind='barh', 
            color=PRIMARY_COLOR,
            ax=ax
        )
        plt.title(f"{metric.title()} Comparison")
        plt.xlabel(metric.title())
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown("### Detailed Model Evaluation")
    model = st.selectbox("Select Model", list(MODEL_METRICS.keys()))
    
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{MODEL_METRICS[model]['accuracy']:.3f}")
    cols[1].metric("Precision", f"{MODEL_METRICS[model]['precision']:.3f}")
    cols[2].metric("Recall", f"{MODEL_METRICS[model]['recall']:.3f}")
    cols[3].metric("F1 Score", f"{MODEL_METRICS[model]['f1']:.3f}")

def show_financial_health(distress_thresh, safe_thresh):
    st.markdown('<h2 class="section-header">Financial Health Assessment</h2>', unsafe_allow_html=True)
    
    if st.session_state.data.empty:
        st.warning("No data available for analysis")
        return
    
    analyzer = ZScoreAnalyzer(st.session_state.data, distress_thresh, safe_thresh)
    zscores = analyzer.calculate()
    
    if not zscores.empty:
        st.markdown("### Z-Score Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.hist(zscores['Z-Score'], bins=30, color=PRIMARY_COLOR)
            ax.axvline(distress_thresh, color=ACCENT_COLOR, linestyle='--')
            ax.axvline(safe_thresh, color=SECONDARY_COLOR, linestyle='--')
            plt.xlabel("Z-Score")
            plt.ylabel("Count")
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            status_dist = zscores['Status'].value_counts(normalize=True)
            fig, ax = plt.subplots(figsize=(6,6))
            status_dist.plot(
                kind='pie', 
                autopct='%1.1f%%',
                colors=[ACCENT_COLOR, '#cccccc', SECONDARY_COLOR],
                ax=ax
            )
            plt.ylabel('')
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown("### Company Status Classification")
        st.dataframe(zscores[['Z-Score', 'Status']].sample(5, random_state=42))

def show_comparative_analysis():
    st.markdown('<h2 class="section-header">Comparative Financial Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.data.empty:
        st.warning("No data available for comparison")
        return
    
    st.markdown("### Key Financial Ratios")
    ratios = {
        'Current Ratio': 'Current Assets / Total Current Liabilities',
        'Debt/Equity': 'Total Liabilities / Market Value',
        'ROA': 'Net Income / Total Assets',
        'Profit Margin': 'Net Income / Net Sales'
    }
    
    selected_ratio = st.selectbox("Select Ratio", list(ratios.keys()))
    
    try:
        numerator, denominator = ratios[selected_ratio].split('/')
        ratio_values = (
            st.session_state.data[numerator.strip()] / 
            st.session_state.data[denominator.strip()]
        )
        
        fig, ax = plt.subplots(figsize=(10,4))
        ratio_values.plot(kind='hist', bins=30, color=PRIMARY_COLOR, ax=ax)
        plt.title(f"Distribution of {selected_ratio}")
        plt.xlabel("Ratio Value")
        st.pyplot(fig)
        plt.close(fig)
    except KeyError as e:
        st.error(f"Missing data for calculation: {str(e)}")
    except Exception as e:
        st.error(f"Error generating analysis: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
