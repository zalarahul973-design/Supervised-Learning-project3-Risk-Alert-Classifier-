PROJECT TITLE:
        
                         Risk Alert Classifier 

 📊 Machine Learning Classification Project

🔍 GIF LINK  
=====check your link=====
gif.gif
🎯 Objectives
The main objectives of this project are:
•	Load and understand the dataset 
•	Separate input features and target variable 
•	Split data into training and testing datasets 
•	Identify missing values 
•	Handle missing numerical values 
•	Encode categorical variables 
•	Analyze class imbalance 
•	Apply sampling techniques 
•	Train Logistic Regression 
•	Train Decision Tree 
•	Train Random Forest 
•	Compare Machine Learning models 
•	Analyze overfitting 
•	Perform RandomizedSearchCV 
•	Perform GridSearchCV 
•	Tune Random Forest hyperparameters 
•	Compare tuned and untuned models 
•	Plot ROC curves 
•	Calculate AUC-ROC 
•	Select the final model 
•	Analyze False Positive and False Negative cases 
•	Provide business recommendations 
________________________________________
📂 Dataset Information
Dataset
Risk_alert_classifier_pro3.csv
Dataset Size
Rows    : 4,600
Columns : 19
Input Features
18
Target
risk_status
________________________________________
🧾 Dataset Columns
Column	Description
customer_id	Unique customer ID
age	Customer age
gender	Customer gender
region	Customer region
employment_type	Type of employment
annual_income_inr	Annual income in INR
credit_score	Customer credit score
credit_utilization_ratio	Credit utilization ratio
missed_payments_12m	Missed payments during last 12 months
avg_late_payment_days	Average late payment days
monthly_transaction_count	Number of monthly transactions
monthly_spend_inr	Monthly spending
cash_advance_count_6m	Cash advances in last 6 months
complaints_last_6m	Customer complaints in last 6 months
failed_login_attempts_3m	Failed login attempts in last 3 months
account_tenure_months	Account tenure in months
last_transaction_date	Date of last transaction
debt_balance_inr	Customer debt balance
risk_status	Target variable
________________________________________
🔵 Q7 - Load Dataset
The dataset is loaded using Pandas.
import pandas as pd

df = pd.read_csv("Risk_alert_classifier_pro3.csv")

X = df.drop("risk_status", axis=1)
y = df["risk_status"]

print("Input Features:")
print(X.columns.tolist())

print("\nTarget Variable:")
print("risk_status")

print("\nInput Shape:", X.shape)
print("Target Shape:", y.shape)
Result
Input Shape: (4600, 18)
Target Shape: (4600,)
The project uses 18 input features and risk_status as the target variable. 
________________________________________
🟢 Q8 - Train Test Split
The dataset is divided into training and testing datasets.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
Dataset Split
Dataset	Shape
Training Features	(3680, 18)
Testing Features	(920, 18)
Training Target	(3680,)
Testing Target	(920,)
Stratification is used so that the class distribution remains approximately consistent between training and testing datasets. 
________________________________________
🟡 Q9 - Missing Value Handling
Missing values are checked before preprocessing.
The notebook identified missing values in several columns including:
•	age 
•	region 
•	employment_type 
•	annual_income_inr 
•	credit_score 
•	credit_utilization_ratio 
•	monthly_spend_inr 
For numerical columns, KNN Imputation is used.
from sklearn.impute import KNNImputer

numeric_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

imputer = KNNImputer()

X_train[numeric_columns] = imputer.fit_transform(
    X_train[numeric_columns]
)

X_test[numeric_columns] = imputer.transform(
    X_test[numeric_columns]
)
After imputation, numerical missing values were filled. Categorical missing values were handled later during preprocessing. 
________________________________________
🟠 Q10 - Logistic Regression
Logistic Regression is used as an initial classification model.
Categorical variables are converted using One-Hot Encoding.
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Logistic Regression Model Successfully Trained!")
print("Predictions:")
print(y_pred[:5])
The Logistic Regression model was successfully trained, although the notebook reported an lbfgs convergence warning. 
________________________________________
🟣 Q11 - Logistic Regression Evaluation
The Logistic Regression model was evaluated using:
•	Accuracy 
•	Precision 
•	Recall 
•	F1 Score 
•	Confusion Matrix 
Confusion Matrix
[[805   4]
 [  7 104]]
Performance
Metric	Score
Accuracy	98.80%
Precision	96.30%
Recall	93.69%
F1 Score	94.98%
The model correctly classified most customers but produced both False Positive and False Negative predictions. 
 
________________________________________
🔴 Q12 - Confusion Matrix
The confusion matrix is used to understand classification errors.
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

TN, FP, FN, TP = cm.ravel()

print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive:", TP)
Result
True Negative  (TN): 805
False Positive (FP): 4
False Negative (FN): 7
True Positive  (TP): 104
Errors
Type-I Error  = False Positive = 4
Type-II Error = False Negative = 7
The notebook identifies False Negative cases as particularly important because they represent high-risk customers incorrectly classified as low-risk. 
________________________________________
📊 Q13 - Class Distribution
The target variable is imbalanced.
print(y.value_counts())

print(
    y.value_counts(normalize=True) * 100
)
Class Distribution
Risk Status	Customers	Percentage
Low Risk 0	4,043	87.89%
High Risk 1	557	12.11%
The High Risk class is the minority class. 
Visualization
A bar chart is created to visualize:
Low Risk (0)
High Risk (1)
 
________________________________________
⚖️ Q14 - Sampling Techniques
Because the dataset is imbalanced, different sampling techniques are applied.
1. Random Under-Sampling
from imblearn.under_sampling import RandomUnderSampler

under_sampler = RandomUnderSampler(
    random_state=42
)

X_under, y_under = under_sampler.fit_resample(
    X_train,
    y_train
)
Result:
0 → 446
1 → 446
________________________________________
2. Random Over-Sampling
from imblearn.over_sampling import RandomOverSampler

over_sampler = RandomOverSampler(
    random_state=42
)

X_over, y_over = over_sampler.fit_resample(
    X_train,
    y_train
)
Result:
0 → 3234
1 → 3234
________________________________________
3. SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    random_state=42
)

X_smote, y_smote = smote.fit_resample(
    X_train,
    y_train
)
Result:
0 → 3234
1 → 3234
________________________________________
4. ADASYN
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(
    random_state=42
)

X_adasyn, y_adasyn = adasyn.fit_resample(
    X_train,
    y_train
)
Result:
0 → 3234
1 → 3226
The notebook compares Under-Sampling, Over-Sampling, SMOTE, and ADASYN. 
________________________________________
📈 Q15 - Balancing Comparison
Logistic Regression is evaluated before and after balancing.
Before Balancing
Metric	Score
Recall	96.40%
F1 Score	98.17%
AUC-ROC	99.996%
Random Over-Sampling
Metric	Score
Recall	100%
F1 Score	98.23%
AUC-ROC	99.998%
SMOTE
Metric	Score
Recall	100%
F1 Score	98.67%
AUC-ROC	99.998%
ADASYN
Metric	Score
Recall	100%
F1 Score	98.67%
AUC-ROC	99.998%
Balancing improved Recall from approximately 96.40% to 100% for the evaluated Logistic Regression experiments. 
________________________________________
🌳 Q16 - Decision Tree
A Decision Tree classifier is trained.
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

y_pred_dt = dt_model.predict(
    X_test
)
Decision Tree Performance
Metric	Low Risk	High Risk
Precision	0.99	0.95
Recall	0.99	0.89
F1 Score	0.99	0.92
Accuracy
98.15%
The Decision Tree achieved 98.15% test accuracy. 
________________________________________
🔍 Q17 - Overfitting Analysis
Training and testing accuracy are compared.
train_pred = dt_model.predict(X_train)
test_pred = dt_model.predict(X_test)

train_accuracy = accuracy_score(
    y_train,
    train_pred
)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)
Result
Training Accuracy: 100%
Testing Accuracy : 98.15%
The notebook's threshold-based check reports:
Model is Generalizing Well
The difference between training and testing accuracy is below the 5% threshold used in the notebook. 
________________________________________
🌲 Q18 - Random Forest
Random Forest is trained to improve classification performance.
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

y_pred_rf = rf_model.predict(
    X_test
)
Random Forest Performance
Metric	Score
Accuracy	99.78%
Precision	100.00%
Recall	98.20%
F1 Score	99.09%
AUC-ROC	99.987%
Random Forest significantly improves upon the Decision Tree. 
________________________________________
⚔️ Q19 - Decision Tree vs Random Forest
The two models are compared using training and testing accuracy.
Decision Tree
Training Accuracy: 100%
Testing Accuracy : 98.15%
Random Forest
Training Accuracy: 100%
Testing Accuracy : 99.78%
Model	Training Accuracy	Testing Accuracy
Decision Tree	100%	98.15%
Random Forest	100%	99.78%
Random Forest performs better on the testing dataset. 
________________________________________
🔎 Q20 - RandomizedSearchCV
RandomizedSearchCV is used to search for better hyperparameters.
Decision Tree Parameters
dt_params = {
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 8],
    "criterion": ["gini", "entropy"]
}
Random Forest Parameters
rf_params = {
    "n_estimators": [100, 150, 200],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}
Both searches use:
n_iter = 10
cv = 5
scoring = "roc_auc"
random_state = 42
Best Decision Tree Parameters
max_depth = 5
min_samples_split = 5
min_samples_leaf = 1
criterion = entropy
Best Decision Tree CV AUC
0.9719268742
Best Random Forest Parameters
n_estimators = 200
max_depth = 15
min_samples_split = 2
min_samples_leaf = 1
max_features = log2
Best Random Forest CV AUC
0.9972254950
Random Forest achieved a substantially higher CV AUC than Decision Tree during RandomizedSearchCV. 
________________________________________
🎯 Q21 - GridSearchCV
GridSearchCV is used for further Random Forest tuning.
grid_params = {
    "n_estimators": [100, 150, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
Best Parameters
n_estimators = 200
max_depth = 10
min_samples_split = 5
min_samples_leaf = 1
Best CV AUC
0.9974513226
Final Tuned Model
RandomForestClassifier(
    max_depth=10,
    min_samples_split=5,
    n_estimators=200,
    random_state=42
)
The tuned model is stored as:
best_model
________________________________________
📊 Q22 - Tuned vs Untuned
The untuned Random Forest is compared with the tuned Random Forest.
Metric	Untuned RF	Tuned RF
Accuracy	99.78%	99.89%
Recall	98.20%	99.10%
Improvement
Accuracy:
99.78% → 99.89%

Recall:
98.20% → 99.10%
Hyperparameter tuning improves both Accuracy and Recall. 
________________________________________
📉 Q23 - ROC Curve
ROC curves are generated for:
•	Decision Tree 
•	Random Forest 
•	Tuned Random Forest 
from sklearn.metrics import roc_curve

dt_prob = dt_model.predict_proba(
    X_test
)[:, 1]

rf_prob = rf_model.predict_proba(
    X_test
)[:, 1]

tuned_prob = best_model.predict_proba(
    X_test
)[:, 1]
ROC Curve plots:
False Positive Rate
        vs
True Positive Rate
 
The ROC curve helps compare the classification ability of the models across different probability thresholds. 
________________________________________
📐 Q24 - AUC ROC Comparison
AUC-ROC is calculated for all three models.
Model	AUC-ROC
Decision Tree	94.286%
Random Forest	99.987%
Tuned Random Forest	99.991%
Exact Values
Decision Tree:
0.9428557111

Random Forest:
0.9998719362

Tuned Random Forest:
0.9999109121
Tuned Random Forest achieved the highest AUC-ROC. 
________________________________________
🏆 Q25 - Final Model Selection
The final models are compared using:
•	Recall 
•	F1 Score 
•	AUC-ROC 
Decision Tree
Recall : 0.8918918919
F1     : 0.9209302326
AUC    : 0.9428557111
Random Forest
Recall : 0.9819819820
F1     : 0.9909090909
AUC    : 0.9998719362
Tuned Random Forest
Recall : 0.9909909910
F1     : 0.9954751131
AUC    : 0.9999109121
Selected Model
Tuned Random Forest
The notebook selects the model with the highest Recall, which is the Tuned Random Forest. 
________________________________________
📊 Q26 - Final Analysis and Reporting
Q26 performs the final comparison of:
Decision Tree
Random Forest
Tuned Random Forest
The following metrics are calculated:
Accuracy
Precision
Recall
F1 Score
AUC-ROC
The final model is assigned as:
best_model_name = "Tuned Random Forest"

final_model = best_model
The final predictions and probabilities are generated:
final_pred = final_model.predict(
    X_test
)

final_prob = final_model.predict_proba(
    X_test
)[:, 1]
The project also generates:
•	Final model comparison 
•	Confusion matrix 
•	Business interpretation 
•	Class imbalance analysis 
•	Performance bar chart 
•	 
The evaluation table is saved as:
final_model_evaluation.csv
________________________________________
📋 Q27 - Final Submission
Q27 creates the final evaluation table.
models = {
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Tuned Random Forest": best_model
}

results = []

for name, model in models.items():

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            pred
        ),
        "Precision": precision_score(
            y_test,
            pred
        ),
        "Recall": recall_score(
            y_test,
            pred
        ),
        "F1 Score": f1_score(
            y_test,
            pred
        ),
        "AUC-ROC": roc_auc_score(
            y_test,
            prob
        )
    })

results_df = pd.DataFrame(results)

print("FINAL EVALUATION TABLE")
print(results_df)
The final submission confirms that the Tuned Random Forest is the best-performing model. 
________________________________________
🏆 Final Model Comparison
Model	Accuracy	Precision	Recall	F1 Score	AUC-ROC
Decision Tree	98.15%	95.19%	89.19%	92.09%	94.29%
Random Forest	99.78%	100.00%	98.20%	99.09%	99.987%
Tuned Random Forest	99.89%	100.00%	99.10%	99.55%	99.991%
The final evaluation table is produced by the notebook and confirms Tuned Random Forest as the best model. 
________________________________________
🥇 Final Model Performance
Tuned Random Forest
Accuracy  : 99.89%
Precision : 100.00%
Recall    : 99.10%
F1 Score  : 99.55%
AUC-ROC   : 99.991%
________________________________________
🔲 Confusion Matrix
The final Tuned Random Forest confusion matrix is:
[[809   0]
 [  1 110]]
Actual / Predicted	Low Risk	High Risk
Low Risk	809	0
High Risk	1	110
Values
True Negative  = 809
False Positive = 0
False Negative = 1
True Positive  = 110
The final model produced only one False Negative and zero False Positives on the test set. 
________________________________________
💼 Business Interpretation
False Positive
A False Positive occurs when:
Actual = Low Risk
Predicted = High Risk
This may result in:
•	Unnecessary risk alerts 
•	Additional customer verification 
•	Unnecessary investigation 
•	Possible rejection of a genuine customer 
The project produced:
False Positive = 0
________________________________________
False Negative
A False Negative occurs when:
Actual = High Risk
Predicted = Low Risk
This is more serious because the business may fail to identify a risky customer.
Possible consequences include:
•	Financial loss 
•	Credit risk 
•	Increased business exposure 
•	Failure to take preventive action 
The project produced:
False Negative = 1
Therefore, Recall is an important metric for this project. 
________________________________________
⚠️ Class Imbalance
The original dataset contains:
Low Risk  = 4043
High Risk = 557
Percentage:
Low Risk  = 87.89%
High Risk = 12.11%
Because High Risk is the minority class, Accuracy alone should not be used to evaluate the model.
Important metrics include:
Recall
F1 Score
AUC-ROC
________________________________________
📌 Key Findings
Finding 1
Random Forest performed better than the basic Decision Tree.
Decision Tree Accuracy = 98.15%

Random Forest Accuracy = 99.78%
Finding 2
Hyperparameter tuning improved Random Forest.
Untuned Accuracy = 99.78%
Tuned Accuracy   = 99.89%
Finding 3
Recall improved:
98.20% → 99.10%
Finding 4
Tuned Random Forest achieved the highest AUC-ROC:
99.991%
Finding 5
The final model produced only one False Negative.
Finding 6
Class imbalance makes Recall, F1 Score and AUC-ROC important evaluation metrics.
These findings are consistent with the final analysis and conclusion in the notebook. 
________________________________________
🛠️ Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Matplotlib
Scikit-learn
Imbalanced-learn
Machine Learning Algorithms
Logistic Regression
Decision Tree
Random Forest
Preprocessing
KNN Imputation
Simple Imputation
One-Hot Encoding
ColumnTransformer
Pipeline
Imbalanced Data Techniques
Random Under-Sampling
Random Over-Sampling
SMOTE
ADASYN
Hyperparameter Tuning
RandomizedSearchCV
GridSearchCV
Evaluation
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
ROC Curve
AUC-ROC
________________________________________
📁 Project Structure
Risk-Alert-Classifier/
│
├── 📓 project3(1).ipynb
│
├── 📄 Risk_alert_classifier_pro3.csv
│
├── 📄 final_model_evaluation.csv
│
├── 📄 README.md
│
└── 📁 images/
    │
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── class_distribution.png
    └── model_comparison.png
________________________________________
⚙️ Installation
Step 1 - Install Python
Install Python 3.x on your system.
Step 2 - Install Required Libraries
Open Command Prompt or PowerShell:
pip install pandas numpy matplotlib scikit-learn imbalanced-learn jupyter
________________________________________
▶️ How to Run the Project
Step 1
Clone or download the project.
Step 2
Open the project folder.
Step 3
Make sure the following files are in the same folder:
project3(1).ipynb
Risk_alert_classifier_pro3.csv
Step 4
Start Jupyter Notebook:
jupyter notebook
Step 5
Open:
project3(1).ipynb
Step 6
Run the notebook cells sequentially from Q7 to Q27.
________________________________________
📊 Generated Outputs
The project generates:
Confusion Matrix
ROC Curve
Class Distribution Chart
Model Comparison Chart
Final Evaluation Table
The final evaluation table is saved as:
final_model_evaluation.csv
________________________________________
🧠 Machine Learning Workflow
                Dataset
                   │
                   ▼
           Load Dataset
                   │
                   ▼
        Feature / Target Split
                   │
                   ▼
          Train-Test Split
                   │
                   ▼
         Missing Value Handling
                   │
                   ▼
          Categorical Encoding
                   │
                   ▼
       Class Imbalance Analysis
                   │
                   ▼
       Sampling / Balancing
                   │
                   ▼
         Logistic Regression
                   │
                   ▼
           Decision Tree
                   │
                   ▼
           Random Forest
                   │
                   ▼
        RandomizedSearchCV
                   │
                   ▼
          GridSearchCV
                   │
                   ▼
         Tuned Random Forest
                   │
                   ▼
         ROC / AUC Analysis
                   │
                   ▼
         Final Model Selection
                   │
                   ▼
          Business Analysis
________________________________________
📚 Learning Outcomes
Through this project, the following Machine Learning concepts were implemented:
•	Dataset loading 
•	Feature engineering basics 
•	Target variable selection 
•	Train-test splitting 
•	Stratified sampling 
•	Missing value analysis 
•	KNN Imputation 
•	Categorical encoding 
•	Logistic Regression 
•	Confusion Matrix 
•	Type-I Error 
•	Type-II Error 
•	Class imbalance 
•	Under-Sampling 
•	Over-Sampling 
•	SMOTE 
•	ADASYN 
•	Decision Tree 
•	Overfitting analysis 
•	Random Forest 
•	Model comparison 
•	RandomizedSearchCV 
•	GridSearchCV 
•	Hyperparameter tuning 
•	ROC Curve 
•	AUC-ROC 
•	Recall-based model selection 
•	Business interpretation 
•	Final model reporting 
________________________________________
🚀 Recommendations
Based on the final analysis:
1.	Use Tuned Random Forest as the final classification model. 
2.	Monitor Recall regularly. 
3.	Pay special attention to False Negative cases. 
4.	Do not rely only on Accuracy because the dataset is imbalanced. 
5.	Monitor Recall, F1 Score and AUC-ROC. 
6.	Retrain the model when sufficient new data becomes available. 
These recommendations are explicitly included in the notebook's final submission. 
________________________________________
🔮 Future Improvements
The project can be further improved by adding:
•	Feature Importance visualization 
•	SHAP Explainability 
•	Cross-validation visualization 
•	Probability threshold tuning 
•	Cost-sensitive learning 
•	Automated ML pipeline 
•	Model serialization using Joblib 
•	Flask API deployment 
•	FastAPI deployment 
•	Streamlit web application 
•	Power BI dashboard 
•	Real-time risk prediction 
•	Model monitoring 
•	Automated model retraining 
________________________________________
🎓 Project Conclusion
The Risk Alert Classifier successfully demonstrates a complete Machine Learning classification workflow.
Three major models were evaluated:
Decision Tree
Random Forest
Tuned Random Forest
Random Forest performed better than the basic Decision Tree, and hyperparameter tuning further improved Random Forest performance.
The final selected model is:
🏆 Tuned Random Forest
with:
Accuracy  = 99.89%
Precision = 100.00%
Recall    = 99.10%
F1 Score  = 99.55%
AUC-ROC   = 99.991%
The final model also produced:
TN = 809
FP = 0
FN = 1
TP = 110
Because the dataset is imbalanced and the High Risk class is the minority class, Recall, F1 Score and AUC-ROC are important for evaluating the model.
The project concludes that the Tuned Random Forest is the strongest model among the evaluated classifiers and can be considered for risk classification with appropriate monitoring and periodic retraining. 
________________________________________
👨‍💻 Author
ZALA RAHULBHAI AMBALAL
Skills Demonstrated
Python
Pandas
NumPy
Machine Learning
Scikit-learn
Data Preprocessing
Classification
Random Forest
Decision Tree
Logistic Regression
SMOTE
ADASYN
Hyperparameter Tuning
Model Evaluation
Data Analysis
________________________________________
⭐ Project Highlights
✅ 4,600 Customer Records
✅ 18 Input Features
✅ Binary Classification
✅ Missing Value Handling
✅ KNN Imputation
✅ Class Imbalance Analysis
✅ SMOTE
✅ ADASYN
✅ Logistic Regression
✅ Decision Tree
✅ Random Forest
✅ RandomizedSearchCV
✅ GridSearchCV
✅ ROC Curve
✅ AUC-ROC
✅ Confusion Matrix
✅ Business Interpretation
✅ Final Model Selection
________________________________________
🏆 Final Result
╔════════════════════════════════════════╗
║       RISK ALERT CLASSIFIER            ║
╠════════════════════════════════════════╣
║ Final Model : Tuned Random Forest      ║
║ Accuracy    : 99.89%                  ║
║ Precision   : 100.00%                 ║
║ Recall      : 99.10%                  ║
║ F1 Score    : 99.55%                  ║
║ AUC-ROC     : 99.991%                 ║
╚════════════════════════════════════════╝



