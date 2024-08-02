import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from flask import Flask, request, jsonify
import joblib

# Load the dataset
file_path = 'CreditCardData.csv'  # Adjust the path according to your Replit file structure
data = pd.read_csv(file_path)

# Inspect the dataset to identify the target column
print(data.columns)

# Assuming 'Fraud' is the target column indicating whether a transaction is fraudulent (1) or not (0)
target_column = 'Fraud'

# Data preprocessing
def preprocess_data(data):
    # Drop missing values
    data.dropna(inplace=True)

    # Extract features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Define the preprocessing for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Transform features
    X = preprocessor.fit_transform(X)

    # Balance the dataset using SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor

X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)

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

# Save the best model and preprocessor for real-time implementation
model_path = 'best_fraud_detection_model.pkl'
preprocessor_path = 'preprocessor.pkl'
joblib.dump(best_model, model_path)
joblib.dump(preprocessor, preprocessor_path)

# Flask app for real-time fraud detection
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_df = pd.DataFrame(data, index=[0])

    # Load the preprocessor
    preprocessor = joblib.load(preprocessor_path)

    # Preprocess the input data
    data_processed = preprocessor.transform(data_df)

    # Load the best model
    model = joblib.load(model_path)

    # Make prediction
    prediction = model.predict(data_processed)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
