import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from flask import Flask, request, jsonify
import joblib

# Load the dataset
file_path = 'online_retail.csv'  # Adjust the path according to your Replit file structure
data = pd.read_csv(file_path)

# Inspect the dataset to identify the target column
print(data.columns)

# Create a dummy target column for demonstration purposes
# In real scenarios, you should replace this with your actual target column
data['IsFraud'] = (data['Quantity'] < 0).astype(int)  # Example: negative quantity as fraud indicator

# Data preprocessing
def preprocess_data(data):
    # Drop missing values
    data.dropna(inplace=True)

    # Drop columns that are not useful for the model
    data.drop(['index', 'InvoiceNo', 'StockCode', 'Description', 'InvoiceDate'], axis=1, inplace=True)

    # Example feature engineering: Create dummy variables for 'Country'
    data = pd.get_dummies(data, columns=['Country'], drop_first=True)

    # 'IsFraud' is the target column in this example
    X = data.drop('IsFraud', axis=1)
    y = data['IsFraud']

    # Balance the dataset using SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

# Train classification models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"{name} - Accuracy: {accuracy}")
    print(f"{name} - Precision: {precision}")
    print(f"{name} - Recall: {recall}")
    print(f"{name} - F1 Score: {f1}")
    print(f"{name} - ROC AUC: {roc_auc}\n")

    if roc_auc > best_score:
        best_score = roc_auc
        best_model = model

print(f"Best model: {best_model}")

# Save the best model for real-time implementation
model_path = 'best_fraud_detection_model.pkl'
joblib.dump(best_model, model_path)

# Flask app for real-time fraud detection
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_df = pd.DataFrame(data, index=[0])

    # Preprocess the input data
    data_processed = scaler.transform(data_df)

    # Make prediction
    prediction = best_model.predict(data_processed)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
