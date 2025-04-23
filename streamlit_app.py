import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="Bankruptcy Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
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
from American companies. The analysis compares multiple machine learning models and their performance metrics.
""")

# Column renaming map
rename_map = {
    "X1":  "Current Assets", "X2":  "Cost of Goods Sold", "X3":  "D&A",
    "X4":  "EBITDA", "X5":  "Inventory", "X6":  "Net Income",
    "X7":  "Total Receivables", "X8":  "Market Value", "X9":  "Net Sales",
    "X10": "Total Assets", "X11": "Total Long-term Debt", "X12": "EBIT",
    "X13": "Gross Profit", "X14": "Total Current Liabilities",
    "X15": "Retained Earnings", "X16": "Total Revenue",
    "X17": "Total Liabilities", "X18": "Total Operating Expenses"
}

@st.cache_data
def load_data():
    possible_paths = [
        'data/american_bankruptcy.csv',
        'american_bankruptcy.csv',
        '../data/american_bankruptcy.csv',
        './american_bankruptcy.csv',
    ]
    for path in possible_paths:
        try:
            st.sidebar.info(f"Trying to load from: {path}")
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Data loaded successfully from {path}")
                if "status_label" in df.columns:
                    vals = df['status_label'].unique()
                    if 'failed' in vals:
                        df['Bankrupt'] = df['status_label'].map({'failed':1,'alive':0})
                    elif 'Bankrupt' in vals:
                        df['Bankrupt'] = df['status_label'].map({'Bankrupt':1,'Alive':0})
                    else:
                        df['Bankrupt'] = df['status_label'].apply(
                            lambda x: 1 if x.lower() in ['failed','bankrupt','distress','default'] else 0
                        )
                    st.sidebar.success("✅ Converted status_label to Bankrupt")
                if "Bankruptcy" in df.columns and "Bankrupt" not in df.columns:
                    df['Bankrupt'] = df['Bankruptcy']
                    st.sidebar.success("✅ Renamed Bankruptcy to Bankrupt")
                if "X1" in df.columns:
                    df = df.rename(columns=rename_map)
                    st.sidebar.success("✅ Renamed X1–X18 to descriptive names")
                return df
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    st.sidebar.error("❌ Failed to load data.")
    st.error("Could not load data. Check file paths.")
    return pd.DataFrame()

# Load data
try:
    data = load_data()
    if not data.empty:
        with st.sidebar.expander("📊 Data Information"):
            st.write(f"Rows: {data.shape[0]}")
            st.write(f"Columns: {data.shape[1]}")
            st.write("Columns: " + ", ".join(data.columns))
            required = [
                'Current Assets','Total Current Liabilities','Retained Earnings',
                'Total Assets','EBIT','Market Value','Total Liabilities','Net Sales'
            ]
            missing = [c for c in required if c not in data.columns]
            if missing:
                st.error("Missing: " + ", ".join(missing))
            else:
                st.success("All required columns present")
except Exception as e:
    st.error(f"Data init error: {e}")
    data = pd.DataFrame()

st.session_state['data_loaded'] = not data.empty

feature_names = [
    "Current Assets","Cost of Goods Sold","D&A","EBITDA","Inventory",
    "Net Income","Total Receivables","Market Value","Net Sales",
    "Total Assets","Total Long-term Debt","EBIT","Gross Profit",
    "Total Current Liabilities","Retained Earnings","Total Revenue",
    "Total Liabilities","Total Operating Expenses"
]

metrics = {
    'Decision Tree':       {'accuracy':0.8925,'precision':0.0589,'recall':0.2404,'f1':0.0947,'auc':0.574,'confusion_matrix':[[10893,1102],[218,69]]},
    'Gradient Boosting':   {'accuracy':0.9761,'precision':0.3846,'recall':0.0348,'f1':0.0639,'auc':0.827,'confusion_matrix':[[11979,16],[277,10]]},
    'Random Forest':       {'accuracy':0.9759,'precision':0.3200,'recall':0.0279,'f1':0.0513,'auc':0.835,'confusion_matrix':[[11978,17],[279,8]]},
    'Logistic Regression': {'accuracy':0.9752,'precision':0.3125,'recall':0.0523,'f1':0.0896,'auc':0.827,'confusion_matrix':[[11962,33],[272,15]]},
    'SVM':                  {'accuracy':0.9765,'precision':0.3333,'recall':0.0070,'f1':0.0137,'auc':0.590,'confusion_matrix':[[11991,4],[285,2]]},
    'KNN':                  {'accuracy':0.9589,'precision':0.1414,'recall':0.1498,'f1':0.1455,'auc':0.695,'confusion_matrix':[[11734,261],[244,43]]}
}

feature_importances = {
    'Decision Tree': {'Retained Earnings':0.072059,'Market Value':0.072055,'Inventory':0.070231,'D&A':0.068246,'Gross Profit':0.067548,'Total Receivables':0.065696,'Current Assets':0.065387,'Total Long-term Debt':0.064578,'Total Assets':0.056883,'Total Current Liabilities':0.055932,'Net Income':0.055526,'Total Liabilities':0.052951,'Cost of Goods Sold':0.051296,'Total Operating Expenses':0.047349,'EBITDA':0.041733,'EBIT':0.041661,'Total Revenue':0.027468,'Net Sales':0.023400},
    'Gradient Boosting': {'Total Long-term Debt':0.115407,'Net Income':0.113170,'Retained Earnings':0.088011,'Market Value':0.083996,'Inventory':0.075858,'Total Operating Expenses':0.071508,'Current Assets':0.068556,'Total Receivables':0.066965,'Gross Profit':0.056605,'D&A':0.045299,'Total Liabilities':0.040103,'EBITDA':0.031667,'EBIT':0.030457,'Net Sales':0.028807,'Cost of Goods Sold':0.028534,'Total Current Liabilities':0.022211,'Total Assets':0.017786,'Total Revenue':0.015061},
    'Random Forest': {'Retained Earnings':0.065674,'Market Value':0.062897,'D&A':0.061341,'Current Assets':0.059910,'Total Receivables':0.059713,'Gross Profit':0.058533,'Total Liabilities':0.057575,'Total Assets':0.057426,'Total Current Liabilities':0.055479,'Inventory':0.054929,'Total Long-term Debt':0.054677,'Net Income':0.053633,'Cost of Goods Sold':0.053133,'EBITDA':0.051601,'EBIT':0.050618,'Total Operating Expenses':0.049919,'Total Revenue':0.046852,'Net Sales':0.046092},
    'Logistic Regression': {'Market Value':1.102307,'Current Assets':0.976875,'Total Current Liabilities':0.500875,'EBIT':0.418057,'Total Long-term Debt':0.366918,'Total Liabilities':0.335098,'EBITDA':0.309482,'Inventory':0.285948,'Total Assets':0.231877,'Gross Profit':0.153693,'Cost of Goods Sold':0.065107,'Total Operating Expenses':0.056967,'Retained Earnings':0.054134,'Total Receivables':0.040750,'Net Income':0.019487,'D&A':0.006214,'Net Sales':0.001644,'Total Revenue':0.001644},
    'SVM': {'Current Assets':0.000147,'Total Receivables':0.000090,'Gross Profit':0.000008,'Total Revenue':0.000000,'Cost of Goods Sold':0.000000,'Net Sales':0.000000,'Total Assets':0.000000,'EBITDA':0.000000,'D&A':0.000000,'Total Operating Expenses':0.000000,'Market Value':-0.000008,'Inventory':-0.000008,'Total Current Liabilities':-0.000008,'Net Income':-0.000016,'EBIT':-0.000016,'Total Long-term Debt':-0.000090,'Total Liabilities':-0.000163,'Retained Earnings':-0.000220},
    'KNN': {'Inventory':0.048982,'D&A':0.048754,'Total Long-term Debt':0.042688,'Gross Profit':0.039603,'Retained Earnings':0.030695,'Total Liabilities':0.023482,'Cost of Goods Sold':0.005708,'EBIT':0.004975,'Total Operating Expenses':0.001930,'Total Revenue':0.001262,'Net Sales':0.001262,'Total Current Liabilities':0.000244,'Current Assets':-0.000090,'Total Receivables':-0.000627,'Total Assets':-0.001449,'EBITDA':-0.001767,'Market Value':-0.002597,'Net Income':-0.004633}
}

roc_curves = {
    'Decision Tree':       {'fpr':[0.0,0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],'tpr':[0.0,0.05,0.1,0.15,0.2,0.24,0.32,0.40,0.5,0.6,0.7,0.8,0.9,1.0],'auc':0.574},
    'Gradient Boosting':   {'fpr':[0.0,0.001,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.4,0.6,0.8,1.0],'tpr':[0.0,0.1,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.9,0.95,0.98,1.0],'auc':0.827},
    'Random Forest':       {'fpr':[0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.4,0.6,0.8,1.0],'tpr':[0.0,0.08,0.2,0.3,0.4,0.5,0.65,0.75,0.85,0.92,0.98,1.0],'auc':0.835},
    'Logistic Regression': {'fpr':[0.0,0.002,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.7,0.9,1.0],'tpr':[0.0,0.1,0.2,0.3,0.4,0.5,0.65,0.75,0.85,0.92,0.98,1.0],'auc':0.827},
    'SVM':                  {'fpr':[0.0,0.0003,0.001,0.005,0.01,0.05,0.1,0.3,0.5,0.7,0.9,1.0],'tpr':[0.0,0.05,0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,0.95,1.0],'auc':0.590},
    'KNN':                  {'fpr':[0.0,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0],'tpr':[0.0,0.05,0.1,0.15,0.2,0.25,0.35,0.45,0.6,0.75,0.9,1.0],'auc':0.695}
}

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Overview","Model Comparison","ROC Curves","Feature Importance","Confusion Matrices","Z-Score Analysis"]
selected_page = st.sidebar.radio("Go to", pages)

# Overview Page
if selected_page == "Overview":
    st.markdown('<p class="sub-header">Overview</p>', unsafe_allow_html=True)
    st.markdown("""
    ### Project Summary
    - **Training**: 1999–2011  
    - **Testing**: 2015–2018  
    - **Features**: 18 financial indicators  
    - **Target**: Bankruptcy (binary)
    
    ### Models
    Decision Tree, GBM, RF, Logistic, SVM, KNN
    
    ### Key Metrics
    Accuracy, Precision, Recall, F1, AUC
    """)
    dfm = pd.DataFrame({
        k: [metrics[k]['accuracy'],metrics[k]['precision'],metrics[k]['recall'],metrics[k]['f1'],metrics[k]['auc']]
        for k in metrics
    }, index=["Accuracy","Precision","Recall","F1 Score","AUC"]).T
    c1,c2,c3 = st.columns(3)
    c1.metric("Best AUC", f"{dfm['AUC'].max():.3f}", dfm['AUC'].idxmax())
    c2.metric("Best F1 Score", f"{dfm['F1 Score'].max():.3f}", dfm['F1 Score'].idxmax())
    c3.metric("Best Recall", f"{dfm['Recall'].max():.3f}", dfm['Recall'].idxmax())
    st.markdown("### Model AUC Comparison")
    aucs = dfm['AUC'].sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(10,6))
    bars=ax.bar(aucs.index,aucs.values,color='#395c40')
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f"{b.get_height():.3f}",ha='center',va='bottom')
    ax.set_ylabel('AUC');ax.set_title('AUC Comparison');plt.xticks(rotation=45);plt.tight_layout()
    st.pyplot(fig)
    if st.session_state['data_loaded']:
        st.markdown("### Dataset Preview")
        cols=[c for c in ['status_label','Bankrupt','year','Current Assets','Total Assets','Net Income','EBIT','Market Value'] if c in data.columns]
        if not cols: cols=list(data.columns[:5])
        st.dataframe(data[cols].head())
        st.markdown("### Dataset Statistics")
        st.write(f"Records: {len(data)}, Features: {len(data.columns)}")
        if 'Bankrupt' in data.columns:
            cnt=data['Bankrupt'].value_counts().rename({1:'Bankrupt',0:'Healthy'})
            # Enhanced spacing
            st.markdown("<div style='margin-top:20px;margin-bottom:20px;'></div>", unsafe_allow_html=True)
            fig,ax=plt.subplots(figsize=(8,4))
            bars=ax.bar(cnt.index,cnt.values,color=['#395c40','#a63603'])
            mx=max(cnt.values);ax.set_ylim(0,mx*1.2);ax.set_ylabel('Count')
            for i,v in enumerate(cnt.values):
                ax.text(i,v+mx*0.05,f"{v}",ha='center',va='bottom',fontsize=14)
            plt.tight_layout();st.pyplot(fig)
            st.markdown("<div style='margin-top:20px;margin-bottom:20px;'></div>", unsafe_allow_html=True)

# Model Comparison Page
elif selected_page == "Model Comparison":
    st.markdown('<p class="sub-header">Model Comparison</p>', unsafe_allow_html=True)
    dfm = pd.DataFrame({
        k: [metrics[k]['accuracy'],metrics[k]['precision'],metrics[k]['recall'],metrics[k]['f1'],metrics[k]['auc']]
        for k in metrics
    }, index=["Accuracy","Precision","Recall","F1 Score","AUC"]).T
    st.markdown("### Metrics")
    st.dataframe(dfm.style.highlight_max(axis=0))
    opts=["Accuracy","Precision","Recall","F1 Score","AUC"]
    sel=st.multiselect("Select metrics",opts,default=["Recall","F1 Score","AUC"])
    if sel:
        fig,axs=plt.subplots(1,len(sel),figsize=(15,5))
        if len(sel)==1: axs=[axs]
        for i,m in enumerate(sel):
            d=dfm.sort_values(m,ascending=False)
            bars=axs[i].bar(d.index,d[m],color='#395c40')
            for b in bars:
                axs[i].text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f"{b.get_height():.3f}",ha='center',va='bottom')
            axs[i].set_title(m);axs[i].set_ylim(0,d[m].max()*1.2);axs[i].tick_params(axis='x',rotation=45)
        plt.tight_layout();st.pyplot(fig)
    st.markdown("### Note on Imbalance")
    st.info("Dataset heavily imbalanced; recall is critical.")

# ROC Curves Page
elif selected_page == "ROC Curves":
    st.markdown('<p class="sub-header">ROC Curves</p>', unsafe_allow_html=True)
    st.markdown("Plots of TPR vs FPR. AUC=1 perfect; 0.5 random.")
    mods=list(metrics.keys())
    sel=st.multiselect("Models",mods,default=mods[:3])
    if sel:
        fig,ax=plt.subplots(figsize=(10,8))
        ax.plot([0,1],[0,1],'--',color='gray',alpha=0.8,label='Random')
        cols=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
        for i,m in enumerate(sel):
            fpr,tpr,auc=roc_curves[m]['fpr'],roc_curves[m]['tpr'],roc_curves[m]['auc']
            ax.plot(fpr,tpr,linewidth=2,color=cols[i%len(cols)],label=f"{m} (AUC={auc:.3f})")
        ax.set_xlabel('FPR');ax.set_ylabel('TPR');ax.set_title('ROC Comparison');ax.legend(loc='lower right');ax.set_xlim(0,1);ax.set_ylim(0,1.05);ax.grid(alpha=0.3)
        plt.tight_layout();st.pyplot(fig)
    st.markdown("### Single Model ROC")
    single=st.selectbox("Model",mods)
    fig,ax=plt.subplots(figsize=(8,6))
    ax.plot([0,1],[0,1],'--',color='gray',alpha=0.8,label='Random')
    fpr,tpr,auc=roc_curves[single]['fpr'],roc_curves[single]['tpr'],roc_curves[single]['auc']
    ax.plot(fpr,tpr,linewidth=2,color='#395c40',label=f"AUC={auc:.3f}")
    ax.set_xlabel('FPR');ax.set_ylabel('TPR');ax.set_title(f'{single} ROC');ax.legend(loc='lower right');ax.set_xlim(0,1);ax.set_ylim(0,1.05);ax.grid(alpha=0.3)
    plt.tight_layout();st.pyplot(fig)

# Feature Importance Page
elif selected_page == "Feature Importance":
    st.markdown('<p class="sub-header">Feature Importance</p>', unsafe_allow_html=True)
    mods=list(feature_importances.keys())
    sel=st.selectbox("Model",mods)
    imps=pd.Series(feature_importances[sel]).sort_values(ascending=False)
    n=st.slider("Top features",5,len(feature_names),10)
    st.markdown(f"### Top {n} for {sel}")
    fig,ax=plt.subplots(figsize=(10,8))
    top=imps.head(n)
    maxv=imps.max()
    bars=ax.barh(top.index[::-1],top.values[::-1],color='#395c40')
    for b in bars:
        w=b.get_width();ax.text(w+maxv*0.05,b.get_y()+b.get_height()/2,f"{w:.3f}",va='center',ha='left')
    ax.set_xlim(0,maxv*1.25);ax.set_xlabel('Importance');ax.set_title(f"{sel} Importance");plt.tight_layout();st.pyplot(fig)
    st.markdown("### Top 5 Across Models")
    cm=["Decision Tree","Gradient Boosting","Random Forest","Logistic Regression"]
    dfc=pd.DataFrame(index=range(1,6),columns=cm)
    for m in cm:
        vals=pd.Series(feature_importances[m]).sort_values(ascending=False)
        for i in range(5): dfc.loc[i+1,m]=vals.index[i]
    st.dataframe(dfc)

# Confusion Matrices Page
elif selected_page == "Confusion Matrices":
    st.markdown('<p class="sub-header">Confusion Matrix</p>', unsafe_allow_html=True)
    mods=list(metrics.keys())
    sel=st.selectbox("Model",mods)
    tn,fp=metrics[sel]['confusion_matrix'][0];fn,tp=metrics[sel]['confusion_matrix'][1]
    acc=(tn+tp)/(tn+fp+fn+tp);prec=tp/(tp+fp) if tp+fp>0 else 0;rec=tp/(tp+fn) if tp+fn>0 else 0;f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0
    st.markdown(f"### {sel}")
    c1,c2=st.columns([2,1])
    with c1:
        df_cm=pd.DataFrame([[tn,fp],[fn,tp]],index=['Actual Alive','Actual Bankrupt'],columns=['Pred Alive','Pred Bankrupt'])
        st.dataframe(df_cm)
        pct=np.array([[100*tn/(tn+fp) if tn+fp>0 else 0,100*fp/(tn+fp) if tn+fp>0 else 0],[100*fn/(fn+tp) if fn+tp>0 else 0,100*tp/(fn+tp) if fn+tp>0 else 0]])
        html=f"""
        <style>
        .cm-box {{padding:20px;text-align:center;margin:5px;font-weight:bold;color:white;}}
        .box-container {{display:grid;grid-template:1fr 1fr/1fr 1fr;gap:10px;margin:20px 0;}}
        .tn {{background:rgba(57,92,64,0.8);}} .fp,.fn {{background:rgba(166,54,3,0.8);}} .tp {{background:rgba(57,92,64,0.8);}}
        </style>
        <div class="box-container">
          <div class="cm-box tn">TN<br>{tn}<br>({pct[0,0]:.1f}%)</div>
          <div class="cm-box fp">FP<br>{fp}<br>({pct[0,1]:.1f}%)</div>
          <div class="cm-box fn">FN<br>{fn}<br>({pct[1,0]:.1f}%)</div>
          <div class="cm-box tp">TP<br>{tp}<br>({pct[1,1]:.1f}%)</div>
        </div>"""
        st.markdown(html, unsafe_allow_html=True)
    with c2:
        st.markdown("#### Metrics")
        st.write(f"- Accuracy: {acc:.4f}")
        st.write(f"- Precision: {prec:.4f}")
        st.write(f"- Recall: {rec:.4f}")
        st.write(f"- F1 Score: {f1:.4f}")
    st.markdown("### Comparison")
    comp=pd.DataFrame(index=mods,columns=["TP","FN","Detect (%)","FA (%)"])
    for m in mods:
        tn0,fp0=metrics[m]['confusion_matrix'][0];fn0,tp0=metrics[m]['confusion_matrix'][1]
        comp.loc[m]=[tp0,fn0,100*tp0/(tp0+fn0) if tp0+fn0>0 else 0,100*fp0/(tn0+fp0) if tn0+fp0>0 else 0]
    st.dataframe(comp.sort_values("Detect (%)",ascending=False))

# Z-Score Analysis Page
elif selected_page == "Z-Score Analysis":
    st.markdown('<p class="sub-header">Altman Z-Score Analysis</p>', unsafe_allow_html=True)
    st.markdown("""
    Z-Score formula: Z = 1.2·T1 + 1.4·T2 + 3.3·T3 + 0.6·T4 + 0.99·T5  
    Zones: Distress (<1.8), Grey (1.8–2.99), Safe (>2.99)
    """)
    if st.session_state['data_loaded'] and not data.empty:
        req=['Current Assets','Total Current Liabilities','Retained Earnings','Total Assets','EBIT','Market Value','Total Liabilities','Net Sales']
        miss=[c for c in req if c not in data.columns]
        if miss:
            st.error("Missing: "+", ".join(miss))
        else:
            if 'Bankrupt' in data.columns:
                st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)
                bc=int(data['Bankrupt'].sum());ac=len(data)-bc
                fig,ax=plt.subplots(figsize=(8,4))
                bars=ax.bar(['Alive','Bankrupt'],[ac,bc],color=['#395c40','#a63603'])
                mx=max(ac,bc);ax.set_ylim(0,mx*1.2);ax.set_ylabel('Count')
                for i,v in enumerate([ac,bc]):ax.text(i,v+mx*0.05,f"{v}",ha='center',va='bottom',fontsize=14)
                plt.tight_layout();st.pyplot(fig);st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)
            def calc_z(df):
                z=pd.DataFrame(index=df.index)
                z['T1']=(df['Current Assets']-df['Total Current Liabilities'])/df['Total Assets']
                z['T2']=df['Retained Earnings']/df['Total Assets']
                z['T3']=df['EBIT']/df['Total Assets']
                z['T4']=df['Market Value']/df['Total Liabilities']
                z['T5']=df['Net Sales']/df['Total Assets']
                z['Z-Score']=1.2*z['T1']+1.4*z['T2']+3.3*z['T3']+0.6*z['T4']+0.99*z['T5']
                z.replace([np.inf,-np.inf],np.nan,inplace=True)
                z['Z-Score'].fillna(z['Z-Score'].median(),inplace=True)
                z['Z-Score Status']=pd.cut(z['Z-Score'],bins=[-np.inf,1.8,2.99,np.inf],labels=['Distress','Grey','Safe'])
                z['Z-Score Prediction']=(z['Z-Score Status']=='Distress').astype(int)
                z['Actual Status']=df['Bankrupt']
                return z
            zdf=calc_z(data)
            zp,za=zdf['Z-Score Prediction'].values,zdf['Actual Status'].values
            zacc=(zp==za).mean();zprec=(zp&za).sum()/zp.sum() if zp.sum()>0 else 0;zrec=(zp&za).sum()/za.sum() if za.sum()>0 else 0;zf1=2*zprec*zrec/(zprec+zrec) if zprec+zrec>0 else 0
            comp=pd.DataFrame({
                'Model':['Altman Z-Score']+list(metrics.keys()),
                'Accuracy':[zacc]+[metrics[m]['accuracy'] for m in metrics],
                'Precision':[zprec]+[metrics[m]['precision'] for m in metrics],
                'Recall':[zrec]+[metrics[m]['recall'] for m in metrics],
                'F1 Score':[zf1]+[metrics[m]['f1'] for m in metrics]
            }).set_index('Model')
            st.markdown("### Z-Score vs ML Models");st.dataframe(comp.style.highlight_max(axis=0))
            ztn=((zp==0)&(za==0)).sum();zfp=((zp==1)&(za==0)).sum();zfn=((zp==0)&(za==1)).sum();ztp=((zp==1)&(za==1)).sum()
            c1,c2=st.columns([2,1])
            with c1:
                st.markdown("#### Z-Score Confusion Matrix")
                st.dataframe(pd.DataFrame([[ztn,zfp],[zfn,ztp]],index=['Actual Alive','Actual Bankrupt'],columns=['Pred Alive','Pred Bankrupt']))
            with c2:
                st.markdown("#### Z-Score Metrics")
                st.write(f"- Accuracy: {zacc:.4f}")
                st.write(f"- Precision: {zprec:.4f}")
                st.write(f"- Recall: {zrec:.4f}")
                st.write(f"- F1 Score: {zf1:.4f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888888; font-size: 0.8em;">
Bankruptcy Prediction Dashboard | Created with Streamlit
</div>
""", unsafe_allow_html=True)
