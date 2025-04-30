# Bankruptcy Prediction Dashboard

##   Team: Hakob Janesian (hakob@bu.edu) and Hitaishi Hitaishi (hitaishi@bu.edu)

## Project Overview

This project presents a comprehensive bankruptcy prediction system that leverages both **traditional financial analysis** and **modern machine learning techniques** to assess corporate financial health. Using the **American Companies Bankruptcy Prediction Dataset** from Kaggle, which contains financial data from approximately 8000 US public companies between 1999-2018, our dashboard provides deep insights into bankruptcy risk factors and model performance.

The system analyzes 18 key financial indicators, including Current Assets, Net Income, EBITDA, and more, to predict bankruptcy outcomes. By combining the time-tested **Altman Z-Score** methodology with cutting-edge machine learning models, we offer a sophisticated platform for financial risk assessment.

## Dataset and Methodology

### Dataset Source
- **Origin**: American Companies Bankruptcy Prediction Dataset (Kaggle: See References)
- **Time Span**: 1999-2018
- **Companies**: ~8000 US public firms
- **Features**: 18 financial indicators (X1-X18) mapped to standard financial metrics
- **Target**: Binary classification (Bankrupt vs Alive)

### Training and Validation
- **Training Period**: 1999-2011
- **Validation Period**: 2012-2014
- **Testing Period**: 2015-2018

## Dashboard Navigation

### Overview Tab
The overview section provides a **holistic view** of bankruptcy prediction performance across all models. Key features include:

- **Performance Summary**: Displays best-performing models for AUC, F1 Score, and Recall
- **Model Comparison**: Visual comparison of all models' AUC scores
- **Quick Insights**: Identifies optimal models for different metrics
- **Dataset Preview**: Shows key financial indicators and bankruptcy status distribution

The standout finding: **Random Forest** achieves the highest AUC (0.838), while the **Altman Z-Score** delivers superior recall (0.282) despite its traditional methodology.

### Dataset Information Tab
This section provides **in-depth exploration** of the dataset structure:

- **Original Format**: Displays raw data with X1-X18 nomenclature
- **Transformed View**: Shows mapped financial metric names
- **Feature Definitions**: Comprehensive financial metric explanations
- **Class Distribution**: Visualizes the significant class imbalance (healthy vs bankrupt companies)

**Key insight**: The dataset exhibits strong class imbalance.

### Altman Z-Score Analysis Tab
Our flagship financial analysis section showcases the **iconic Altman Z-Score** methodology:

#### The Z-Score Formula
Z = 1.2×T1 + 1.4×T2 + 3.3×T3 + 0.6×T4 + 0.99×T5

Where:
- **T1**: Working Capital / Total Assets
- **T2**: Retained Earnings / Total Assets
- **T3**: EBIT / Total Assets
- **T4**: Market Value / Total Liabilities
- **T5**: Sales / Total Assets

#### Classification Zones
- **Safe Zone** (Z > 2.99): Low bankruptcy risk
- **Grey Zone** (1.8 < Z < 2.99): Moderate risk
- **Distress Zone** (Z < 1.8): High bankruptcy risk

The analysis reveals the Z-Score's **remarkable performance** despite its simplicity:
- **Accuracy**: 88.9%
- **Recall**: 29.3%
- **Precision**: 6.8%
- **F1 Score**: 11.0%

#### Financial Insights
The Altman Z-Score excels in bankruptcy detection through its **balanced approach** to financial health assessment. It combines:
- **Liquidity**: Working capital ratio
- **Profitability**: EBIT and retained earnings
- **Leverage**: Market value to debt ratio
- **Activity**: Asset turnover

### Model Comparison Tab
This section provides **comprehensive benchmarking** of all models:

- **Performance Metrics**: Accuracy, Precision, Recall, F1 Score, and AUC
- **Visual Comparisons**: Interactive charts for metric-specific analysis
- **Traditional vs ML**: Highlights the trade-offs between interpretability and performance

**Key finding**: ML models like Random Forest achieve higher AUC (0.838), but the Altman Z-Score model offers superior bankruptcy detection (recall) at 28.2%.

### ROC Curves Tab
Visualizes **model discrimination ability** through ROC analysis:

- **Combined ROC Plot**: Overlays all models for direct comparison
- **Individual Analysis**: Detailed view of each model's ROC curve
- **AUC Comparison**: Quantifies overall model performance

The Random Forest and Gradient Boosting models show **superior discrimination** with AUCs of 0.838 and 0.827 respectively, while the Altman Z-Score model achieves a respectable 0.592.

### Feature Importance Tab
Reveals **critical financial drivers** for bankruptcy prediction:

- **Model-specific Rankings**: Shows which financial metrics drive each model
- **Top Features**: Retained Earnings, Market Value, and Total Long-term Debt consistently emerge as key indicators
- **Cross-model Comparison**: Highlights agreement and divergence in feature importance

**Insight**: The consistency between ML models and the Altman Z-Score model's component ratios validates the financial theory underlying bankruptcy prediction.

### Confusion Matrices Tab
Provides **detailed classification performance** analysis:

- **Visual Representation**: Color-coded confusion matrices
- **Metric Calculations**: True/False Positive/Negative breakdowns
- **Detection Rates**: Comparative analysis of bankruptcy identification

The Z-Score model achieves the **highest detection rate** at 29.3%, identifying more actual bankruptcies than sophisticated ML models despite higher false positives.

## Key Findings and Implications

### The Enduring Power of the Altman Z-Score model
Despite being developed in 1968, the Altman Z-Score model demonstrates **remarkable effectiveness** in modern bankruptcy prediction. Its strengths include:

1. **Superior Recall**: Highest bankruptcy detection rate among all models
2. **Interpretability**: Clear financial rationale for predictions
3. **Simplicity**: Easy to calculate and communicate to stakeholders
4. **Regulatory Acceptance**: Widely recognized by financial institutions

### Machine Learning Advantages
Modern ML approaches offer complementary benefits:

1. **Higher AUC**: Better overall discrimination ability
2. **Adaptability**: Can incorporate more features and complex relationships
3. **Industry Specificity**: Potential for sector-tailored models
4. **Dynamic Updating**: Ability to retrain with new data

### Practical Implications
For financial practitioners, this analysis suggests a **hybrid approach**:
- Use the Altman Z-score for initial screening and stakeholder communication
- Deploy ML models for refined risk assessment and portfolio management
- Combine both approaches for robust bankruptcy prediction

## Bankruptcy Trends Over Time

We conducted a **yearly trend analysis** of both failure counts and bankruptcy rates (1999–2018):

- **Failed companies** declined from **380** in 1999 to **36** in 2018, peaking at **415** in 2003.
- **Bankruptcy rate** dropped from **7.16%** in 1999 to **1.32%** in 2018, with an average of **6.28%**, and a high of **9.40%** in 2003.

This demonstrates a **steady downward trend** in bankruptcy incidence over the 20 years.

## Technical Implementation

The dashboard leverages Streamlit for interactive visualization, presenting complex financial analytics in an accessible format. Users can explore:
- Model performance across multiple metrics
- Feature importance and its financial implications
- Detailed confusion matrices for error analysis
- Interactive ROC curves and AUC comparisons

## Conclusion

This project demonstrates that **traditional financial analysis and modern machine learning are complementary**, not competitive, in bankruptcy prediction. The Altman Z-Score's enduring relevance, combined with the enhanced discrimination of ML models, provides financial analysts with a powerful toolkit for risk assessment.

Our findings reaffirm the critical role of fundamental financial ratios in bankruptcy prediction while highlighting opportunities for enhanced accuracy through machine learning approaches.

## Reference
The dataset used in this project came from Kaggle, and this is the link:
https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset
