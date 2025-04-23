"""
Configuration settings for the Bankruptcy Prediction Dashboard.
"""

# App settings
APP_TITLE = "Bankruptcy Prediction Dashboard"
APP_ICON = "📊"
APP_DESCRIPTION = "A comprehensive analysis of bankruptcy prediction models using financial data."
GITHUB_REPO = "https://github.com/yourusername/bankruptcy-prediction-dashboard"
APP_AUTHOR = "Your Name"
APP_VERSION = "1.0.0"

# Data settings
DEFAULT_DATA_PATH = "american_bankruptcy.csv"
TRAIN_YEARS = (1999, 2011)
TEST_YEARS = (2015, 2018)

# Feature settings
FEATURE_NAMES = [
    "Current Assets",
    "Cost of Goods Sold",
    "D&A",
    "EBITDA",
    "Inventory",
    "Net Income",
    "Total Receivables",
    "Market Value",
    "Net Sales",
    "Total Assets",
    "Total Long-term Debt",
    "EBIT",
    "Gross Profit",
    "Total Current Liabilities",
    "Retained Earnings",
    "Total Revenue",
    "Total Liabilities",
    "Total Operating Expenses"
]

# Feature descriptions for tooltips
FEATURE_DESCRIPTIONS = {
    "Current Assets": "Resources that can be converted to cash within one year",
    "Cost of Goods Sold": "Direct costs attributable to the production of goods sold",
    "D&A": "Depreciation and Amortization expenses",
    "EBITDA": "Earnings Before Interest, Taxes, Depreciation, and Amortization",
    "Inventory": "Goods available for sale or raw materials",
    "Net Income": "Total earnings or profit after all expenses",
    "Total Receivables": "Money owed to a company by its debtors",
    "Market Value": "Total value of a company's outstanding shares",
    "Net Sales": "Gross sales minus returns, allowances, and discounts",
    "Total Assets": "Sum of all assets owned by a company",
    "Total Long-term Debt": "Loans and financial obligations lasting over one year",
    "EBIT": "Earnings Before Interest and Taxes",
    "Gross Profit": "Net sales minus the cost of goods sold",
    "Total Current Liabilities": "Debts or obligations due within one year",
    "Retained Earnings": "Cumulative net income that is retained for future use",
    "Total Revenue": "Income generated from all business activities",
    "Total Liabilities": "Sum of all current and long-term liabilities",
    "Total Operating Expenses": "Costs associated with day-to-day business operations"
}

# Model settings
MODEL_CONFIGS = {
    "Decision Tree": {
        "color": "#1f77b4",
        "params": {
            "random_state": 42,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }
    },
    "Gradient Boosting": {
        "color": "#ff7f0e",
        "params": {
            "random_state": 42,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3
        }
    },
    "Random Forest": {
        "color": "#2ca02c",
        "params": {
            "random_state": 42,
            "n_estimators": 100,
            "max_features": "sqrt",
            "max_depth": None
        }
    },
    "Logistic Regression": {
        "color": "#d62728",
        "params": {
            "random_state": 42,
            "solver": "liblinear",
            "C": 1.0,
            "class_weight": "balanced"
        }
    },
    "SVM": {
        "color": "#9467bd",
        "params": {
            "random_state": 42,
            "kernel": "rbf",
            "C": 1.0,
            "probability": True
        }
    },
    "KNN": {
        "color": "#8c564b",
        "params": {
            "n_neighbors": 5,
            "weights": "uniform",
            "p": 2
        }
    }
}

# Visualization settings
PLOT_STYLE = "seaborn-v0_8-darkgrid"
COLOR_SCHEME = {
    "primary": "#395c40",
    "secondary": "#a63603",
    "tertiary": "#4c4c4c",
    "success": "#198754",
    "danger": "#dc3545",
    "warning": "#ffc107",
    "info": "#0dcaf0"
}
FONT_SIZE = {
    "title": 16,
    "subtitle": 14,
    "axis_label": 12,
    "tick_label": 10,
    "annotation": 8
}
FIG_SIZE = {
    "small": (6, 4),
    "medium": (8, 6),
    "large": (10, 8),
    "wide": (12, 6),
    "tall": (6, 10)
}

# UI settings
SIDEBAR_WIDTH = 300
DEFAULT_PADDING = 1
SECTION_SPACING = 0.5
USE_CUSTOM_CSS = True
CUSTOM_CSS = """
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
.metric-container {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.footer {
    text-align: center;
    color: #6c757d;
    padding: 20px 0;
    font-size: 0.8rem;
}
.feature-card {
    border: 1px solid #e9ecef;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 15px;
}
.feature-card:hover {
    background-color: #f8f9fa;
    transition: background-color 0.3s;
}
.highlight {
    background-color: rgba(57, 92, 64, 0.1);
    padding: 2px 4px;
    border-radius: 3px;
}
.important-value {
    font-weight: bold;
    color: #395c40;
}
.warning-value {
    font-weight: bold;
    color: #a63603;
}
"""

# Risk analysis settings
BANKRUPTCY_RISK_LEVELS = {
    "Low": {
        "threshold": 0.2,
        "color": "#198754",
        "description": "Companies with minimal risk of bankruptcy in the near future."
    },
    "Medium": {
        "threshold": 0.5,
        "color": "#ffc107",
        "description": "Companies showing some warning signs but not immediate danger."
    },
    "High": {
        "threshold": 0.7,
        "color": "#dc3545",
        "description": "Companies with significant financial distress indicators."
    },
    "Very High": {
        "threshold": 0.9,
        "color": "#6c040c",
        "description": "Companies with severe financial distress requiring immediate attention."
    }
}

# Financial ratio descriptions
FINANCIAL_RATIOS = {
    "Current Ratio": {
        "formula": "Current Assets / Total Current Liabilities",
        "description": "Measures a company's ability to pay short-term obligations.",
        "good_range": "1.5 to 3.0",
        "warning_threshold": 1.0
    },
    "Debt to Equity": {
        "formula": "Total Liabilities / (Total Assets - Total Liabilities)",
        "description": "Indicates the proportion of equity and debt used to finance assets.",
        "good_range": "1.0 to 1.5",
        "warning_threshold": 2.0
    },
    "Return on Assets": {
        "formula": "Net Income / Total Assets",
        "description": "Indicates how profitable a company is relative to its total assets.",
        "good_range": "> 5%",
        "warning_threshold": 0.0
    },
    "Profit Margin": {
        "formula": "Net Income / Total Revenue",
        "description": "Measures the percentage of revenue that exceeds costs.",
        "good_range": "> 10%",
        "warning_threshold": 0.0
    },
    "Asset Turnover": {
        "formula": "Total Revenue / Total Assets",
        "description": "Measures how efficiently a company uses its assets to generate sales.",
        "good_range": "> 0.5",
        "warning_threshold": 0.3
    }
}

# Cache settings
CACHE_TTL = 3600  # Cache time to live in seconds (1 hour)
ENABLE_CACHING = True

# Debug settings
DEBUG_MODE = False
LOG_LEVEL = "INFO"
PROFILE_PERFORMANCE = False
