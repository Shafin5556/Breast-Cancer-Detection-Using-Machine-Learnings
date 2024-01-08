# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Function to print a header
def print_header(title):
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80 + "\n")

# Function for printing model results in a structured way
def print_model_results(model_name, y_test, y_pred, model, X, y, skf):
    print_header(f"{model_name} Model Results")

    print(f"{'Metric':<20}{'Value':<60}")
    print("-" * 80)
    print(f"{'Training Accuracy':<20}{model.score(X_train, y_train):<60.4f}")
    print(f"{'Test Accuracy':<20}{model.score(X_test, y_test):<60.4f}")
    print("-" * 80)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    cv_scores = cross_val_score(model, X, y, cv=skf)
    print(f"\nCross-Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {np.mean(cv_scores):.4f}")

# Data Loading and Preprocessing
data = pd.read_csv('data.csv')  # Replace with your file path
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log_model = LogisticRegression(max_iter=10000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
skf = StratifiedKFold(n_splits=5)
print_model_results("Logistic Regression", y_test, log_pred, log_model, X, y, skf)

# ROC Curve - Logistic Regression
y_scores = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc_score(y_test, y_scores))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC')
plt.legend(loc="lower right")
plt.show()

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print_model_results("Random Forest", y_test, rf_pred, rf_model, X, y, skf)

# ROC Curve - Random Forest
y_scores_rf = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_scores_rf)
plt.figure()
plt.plot(fpr_rf, tpr_rf, label='ROC Curve (area = %0.2f)' % roc_auc_score(y_test, y_scores_rf))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC')
plt.legend(loc="lower right")
plt.show()

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print_model_results("Decision Tree", y_test, dt_pred, dt_model, X, y, skf)

# ROC Curve - Decision Tree
y_scores_dt = dt_model.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_scores_dt)
plt.figure()
plt.plot(fpr_dt, tpr_dt, label='ROC Curve (area = %0.2f)' % roc_auc_score(y_test, y_scores_dt))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree ROC')
plt.legend(loc="lower right")
plt.show()


# ... (existing code remains the same)

# Function to plot and save ROC curve
def plot_and_save_roc_curve(fpr, tpr, roc_auc, model_name):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_name}_ROC.png')  # Save ROC curve as an image
    plt.show()

# ... (existing code remains the same)

# ROC Curve - Logistic Regression and save
y_scores = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)
plot_and_save_roc_curve(fpr, tpr, roc_auc, "Logistic Regression")

# ... (similar changes for other models)

# ROC Curve - Random Forest and save
y_scores_rf = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_scores_rf)
roc_auc_rf = roc_auc_score(y_test, y_scores_rf)
plot_and_save_roc_curve(fpr_rf, tpr_rf, roc_auc_rf, "Random Forest")

# ROC Curve - Decision Tree and save
y_scores_dt = dt_model.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_scores_dt)
roc_auc_dt = roc_auc_score(y_test, y_scores_dt)
plot_and_save_roc_curve(fpr_dt, tpr_dt, roc_auc_dt, "Decision Tree")
