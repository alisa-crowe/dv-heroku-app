# Step 1: Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump


# Step 2: Load the Data
df = pd.read_csv("v11NumericIncidentPrediction.csv")  # Update this path as needed


# Step 3: Preprocessing
df.dropna(inplace=True)


# Identify categorical and numerical columns (excluding Zip Code)
categorical_columns = ['Overall Race', 'City', 'Day of Week', 'Month']
numerical_columns = ['Victim Age', 'Hour']  # Zip Code removed from here


# Create a column transformer with OneHotEncoder for categorical data and StandardScaler for numerical data
preprocessing_pipeline = ColumnTransformer(
   transformers=[
       ('num', StandardScaler(), numerical_columns),
       ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
   ])


# Convert 'Domestic Violence Incident' column to binary
df['Domestic Violence Incident'] = df['Domestic Violence Incident'].astype(int)


# Feature Selection
X = df[numerical_columns + categorical_columns]
y = df['Domestic Violence Incident']


# Step 4: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 5: Combine preprocessing and model training into a single pipeline
model_pipeline = Pipeline(steps=[
   ('preprocessor', preprocessing_pipeline),
   ('classifier', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20, class_weight={0: 2, 1: 3}))
])


# Step 6: Fit the model pipeline
model_pipeline.fit(X_train, y_train)


# Step 7: Make Predictions and Evaluate the Model
y_pred = model_pipeline.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Step 8: Save the Entire Model Pipeline
dump(model_pipeline, 'domestic_violence_model_pipeline.pkl', compress=3)
print("Model pipeline saved as 'domestic_violence_model_pipeline.pkl'.")
