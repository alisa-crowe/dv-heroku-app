# Step 1: Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Step 2: Load the Data
df = pd.read_csv("v11NumericIncidentPrediction.csv")  # Update this path as needed


# Step 3: Preprocessing
# Handle missing values if any (for simplicity, we'll drop them)
df.dropna(inplace=True)


# Convert categorical columns to numerical using LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = ['Agency', 'CIBRS Offense Code', 'CIBRS Offense Description', 'Victim Category',
                      'Overall Race', 'City', 'HHSA Region', 'Day of Week']
for col in categorical_columns:
   df[col] = label_encoder.fit_transform(df[col])


# Convert 'Domestic Violence Incident' column to binary (True/False -> 1/0)
df['Domestic Violence Incident'] = df['Domestic Violence Incident'].astype(int)


## Step 4: Feature Selection
features = ['Victim Age', 'Overall Race', 'City', 'Zip Code', 'Hour', 'Day of Week', 'Month']
X = df[features]
y = df['Domestic Violence Incident']


# Step 5: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 6: Standardize the Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 7: Train the Random Forest with Custom Class Weights
dict_weights = {0: 3, 1: 2}
weighted_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20, class_weight=dict_weights)
weighted_model.fit(X_train, y_train)


# Step 8: Make Predictions with the Weighted Model
y_prob_weighted = weighted_model.predict_proba(X_test)[:, 1]  # Probabilities for class '1' (domestic violence)


# Step 9: Adjust the Threshold to Minimize False Negatives
threshold = 0.4
y_pred_adjusted = (y_prob_weighted >= threshold).astype(int)


# Step 10: Evaluate the Adjusted Model
print("Confusion Matrix (Adjusted Threshold Model):")
conf_matrix_adjusted = confusion_matrix(y_test, y_pred_adjusted)
print(conf_matrix_adjusted)


# Plot the confusion matrix for the adjusted model
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_adjusted, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Adjusted Threshold Model')
plt.show()


# Classification Report for the Adjusted Model
print("\nClassification Report (Adjusted Threshold Model):")
print(classification_report(y_test, y_pred_adjusted))


# Step 11: Plot Feature Importances
feature_importances = pd.Series(weighted_model.feature_importances_, index=features)
plt.figure(figsize=(12, 8))
feature_importances.sort_values().plot(kind='barh')
plt.title('Feature Importances in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

compressed_model_path = 'domestic_violence_model_compressed.pkl'
joblib.dump(weighted_model, compressed_model_path, compress=3)