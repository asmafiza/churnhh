This project implements:
*   **Data Preprocessing**: Handling missing values, casting structured types (`TotalCharges`), and feature encoding.
*   **Exploratory Data Analysis (EDA)**: Class imbalance evaluation and feature correlation analysis.
*   **Machine Learning Modeling**: Comparative evaluation of Logistic Regression, Decision Trees, Random Forests, K-Nearest Neighbors, and Naive Bayes classifiers.

---

## 📊 Dataset Structure
The dataset source contains **7,043 rows** and **21 columns**. Key attributes include:
*   `customerID`: Unique identifier (dropped during preprocessing)
*   `gender`, `SeniorCitizen`, `Partner`, `Dependents`: Demographic profiles
*   `tenure`: Months the customer has stayed with the company
*   `PhoneService`, `MultipleLines`, `InternetService`, etc.: Signed up services
*   `Contract`, `PaperlessBilling`, `PaymentMethod`: Account configuration terms
*   `MonthlyCharges`, `TotalCharges`: Financial vectors
*   `Churn`: Target labels (Yes/No indicating whether the customer churned)

---

## 🛠️ Architecture & Dependencies
The core analysis relies heavily on the following software packages:
*   `pandas` & `numpy` - Data Manipulation
*   `matplotlib` & `seaborn` - Matrix Plotting & Distributions
*   `scikit-learn` - Machine Learning & Metrics Pipelines

---

## 📈 Key Visualizations Included
1.  **Customer Churn Distribution**: Bar charts showing target variable balances.
2.  **Correlation Heatmap**: Feature correlation matrix showcasing multi-collinearity configurations across features.

---

## 🚀 Execution & Modeling
The pipeline executes metrics comparison tracking for:
*   Accuracy
*   Precision
*   Recall
*   F1-Score
*   Confusion Matrix Evaluation
