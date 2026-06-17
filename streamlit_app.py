import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# --- 1. PAGE SETUP & CONFIG ---
st.set_page_config(page_title="Telco Churn Analytics Dashboard", layout="wide")
st.title("📊 Telco Customer Churn - End-to-End Analytics Dashboard")
st.markdown("Yeh dashboard aapke data ki cleaning, EDA, Supervised ML Comparison, Unsupervised Clustering, aur **Live Customer Prediction** ko ek sath screen par show karta hai.")
st.markdown("---")

# --- 2. DATA LOADING & CLEANING (FOOLPROOF ENGINE) ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
        
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    
    if 'Churn' in df.columns:
        df['Churn_Numeric'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1'] else 0)
    
    X_raw = df.drop(columns=['Churn', 'Churn_Numeric'], errors='ignore')
    df_encoded = pd.get_dummies(X_raw, drop_first=True)
    df_encoded['Churn'] = df['Churn_Numeric']
    df_encoded = df_encoded.astype(float)
            
    return df, df_encoded

try:
    df, df_encoded = load_and_clean_data()
    
    # KPI Row (Main Numbers)
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(label="Total Customers (Rows)", value=df.shape[0])
    with kpi2:
        st.metric(label="Average Monthly Charges", value=f"${df['MonthlyCharges'].mean():.2f}")
    with kpi3:
        churn_rate = (df['Churn'].value_counts(normalize=True).get('Yes', 0)) * 100
        st.metric(label="Overall Churn Rate", value=f"{churn_rate:.1f}%")
        
    st.markdown("---")
    
    # --- 3. EXPLORATORY DATA ANALYSIS (EDA) ---
    st.header("📈 1. Exploratory Data Analysis (EDA)")
    eda_col1, eda_col2, eda_col3 = st.columns(3)
    
    with eda_col1:
        st.subheader("Customer Churn Distribution")
        fig_churn, ax_churn = plt.subplots(figsize=(5, 4))
        sns.countplot(x="Churn", data=df, ax=ax_churn, palette="Set1")
        st.pyplot(fig_churn)
        
    with eda_col2:
        st.subheader("Distribution of Tenure")
        fig_tenure, ax_tenure = plt.subplots(figsize=(5, 4))
        sns.histplot(df['tenure'], ax=ax_tenure, color="skyblue", kde=True)
        st.pyplot(fig_tenure)
        
    with eda_col3:
        st.subheader("Correlation Heatmap")
        fig_heat, ax_heat = plt.subplots(figsize=(5, 4))
        sns.heatmap(df_encoded.corr(numeric_only=True), cmap="coolwarm", ax=ax_heat, cbar=False)
        st.pyplot(fig_heat)
        
    st.markdown("---")
    
    # --- 4. SUPERVISED LEARNING (MODEL COMPARISON) ---
    st.header("🤖 2. Supervised Learning - Model Comparison")
    
    X = df_encoded.drop(columns=['Churn'], errors='ignore')
    y = df_encoded["Churn"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }
    
    results = []
    confusion_matrices = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, pred, average='macro', zero_division=0)
        
        results.append([name, acc, prec, rec, f1])
        confusion_matrices[name] = confusion_matrix(y_test, pred)
        
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
    
    ml_col1, ml_col2 = st.columns([1, 1])
    with ml_col1:
        st.subheader("Models Performance Table")
        st.dataframe(results_df.style.format({"Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1": "{:.2%}"}))
        
    with ml_col2:
        st.subheader("Accuracy Comparison Graph")
        fig_bar, ax_bar = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x="Model", y="Accuracy", data=results_df, ax=ax_bar, palette="viridis")
        plt.xticks(rotation=20)
        st.pyplot(fig_bar)
        
    st.subheader("🔍 Confusion Matrix Explorer")
    selected_model_name = st.selectbox("Kisi bhi model ki Confusion Matrix dekhne ke liye select karein:", list(models.keys()))
    
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(confusion_matrices[selected_model_name], annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix - {selected_model_name}")
    st.pyplot(fig_cm)
    
    st.markdown("---")
    
    # --- 5. UNSUPERVISED LEARNING (CLUSTERING) ---
    st.header("🎯 3. Unsupervised Learning - Customer Segmentation")
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters
    
    cluster_col1, cluster_col2 = st.columns(2)
    with cluster_col1:
        st.subheader("K-Means Customer Segmentation Graph")
        fig_cluster, ax_cluster = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df["tenure"], y=df["MonthlyCharges"], hue=df["Cluster"], palette="Set1", ax=ax_cluster)
        st.pyplot(fig_cluster)
        
    with cluster_col2:
        st.subheader("Hierarchical Clustering (Dendrogram - Sample 500 rows)")
        linked = linkage(X_scaled[:500], method="ward")
        fig_dendro, ax_dendro = plt.subplots(figsize=(6, 4))
        dendrogram(linked, ax=ax_dendro, no_labels=True)
        st.pyplot(fig_dendro)
        
    st.subheader("📋 Cluster Profile (Mean Analysis)")
    st.dataframe(df.groupby("Cluster").mean(numeric_only=True))
    
    st.markdown("---")
    
    # --- 6. BUSINESS INSIGHTS ---
    st.header("💡 4. Strategic Business Insights")
    ins1, ins2 = st.columns(2)
    with ins1:
        st.info("**1. High Churn Risk Identification:**\n\nData ke mutabiq shuruati months (low tenure) wale aur high Monthly Charges wale customers sab se zyada churn kar rahe hain.")
        st.success("**2. Targeted Discounts:**\n\nRisky clusters ko churn hone se bachane ke liye unhein customized packages aur promo discounts offer kiye ja sakte hain.")
    with ins2:
        st.warning("**3. Marketing Campaigns Using Clusters:**\n\nK-Means ke banaye gaye clusters ko use kar ke marketing team har type ke customer group ko target kar sakti hai.")
        st.error("**4. Retention Strategy:**\n\nHigh tenure wale loyal customers ko long-term contracts ke zariye retain rakhna company ke liye sab se behtar hai.")

    # --- 7. LIVE CHURN PREDICTION SYSTEM ---
    st.markdown("---")
    st.header("🔮 5. Live Customer Churn Predictor")
    st.markdown("Niche diye gaye fields mein customer ka data enter karein aur check karein ke woh company chor kar ja sakta hai ya nahi:")

    # Background mein model ko dobara train kar ke save kar rahe hain taake live prediction chalay
    X_rf = df_encoded[["tenure", "MonthlyCharges"]]
    y_rf = df_encoded["Churn"]
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_rf, y_train_rf)
    joblib.dump(rf, "random_forest_model.pkl")

    # Inputs for Prediction
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        input_tenure = st.number_input("Customer ka Tenure (Months) likhein:", min_value=0, max_value=100, value=12)
    with p_col2:
        input_charges = st.number_input("Monthly Charges ($) likhein:", min_value=0.0, max_value=200.0, value=50.0)

    if st.button("🚀 Churn Predict Karein"):
        loaded_model = joblib.load("random_forest_model.pkl")
        user_features = pd.DataFrame([[input_tenure, input_charges]], columns=["tenure", "MonthlyCharges"])
        
        prediction = loaded_model.predict(user_features)[0]
        prediction_proba = loaded_model.predict_proba(user_features)[0][1]
        
        st.markdown("### **Prediction Result:**")
        if prediction == 1:
            st.error(f"⚠️ **High Risk!** Yeh customer company chor kar **ja sakta hai** (Churn Risk).")
            st.write(f"📊 **Churn Probability:** {prediction_proba * 100:.1f}% chances hain.")
        else:
            st.success(f"✅ **Safe!** Yeh customer loyal lag raha hai, iske company chorne ka **koi khatra nahi hai**.")
            st.write(f"📊 **Churn Probability:** Only {prediction_proba * 100:.1f}% chances hain.")

except Exception as e:
    st.error(f"App ko run karne mein masla aa raha hai: {e}")
