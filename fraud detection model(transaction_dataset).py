import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import numpy as np

# Load the dataset
file_path = 'transaction_dataset.csv'
data = pd.read_csv(file_path)

# Display the columns of the dataset
print(data.columns)

# Check for non-numeric values in the dataset
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"Column '{col}' contains non-numeric values.")

# Preprocess the data
def preprocess_data(data):
    # Define the target and features
    target_column = 'FLAG'
    categorical_features = ['Address', ' ERC20 most sent token type', ' ERC20_most_rec_token_type']
    numerical_features = data.drop(columns=[target_column] + categorical_features).select_dtypes(include=[np.number]).columns.tolist()

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply preprocessing
    X = preprocessor.fit_transform(X)

    # Handle imbalanced data using SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f'{model.__class__.__name__} - Accuracy: {accuracy}')
    print(f'{model.__class__.__name__} - Precision: {precision}')
    print(f'{model.__class__.__name__} - Recall: {recall}')
    print(f'{model.__class__.__name__} - F1 Score: {f1}')
    print(f'{model.__class__.__name__} - ROC AUC: {roc_auc}')

# Preprocess the data
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)

# Train and evaluate Logistic Regression
lr_model = LogisticRegression(max_iter=1000, solver='lbfgs')
lr_model.fit(X_train, y_train)
evaluate_model(lr_model, X_test, y_test)

# Train and evaluate Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
evaluate_model(dt_model, X_test, y_test)

# Train and evaluate Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
evaluate_model(rf_model, X_test, y_test)

# Train and evaluate Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
evaluate_model(gb_model, X_test, y_test)

# Determine the best model based on F1 Score or other metrics
best_model = max((lr_model, dt_model, rf_model, gb_model), key=lambda model: f1_score(y_test, model.predict(X_test)))
print(f'Best model: {best_model.__class__.__name__}')

# Save the best model (optional)
import joblib
joblib.dump(best_model, 'best_fraud_detection_model.pkl')
