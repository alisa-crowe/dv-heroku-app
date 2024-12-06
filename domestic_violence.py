# Step 1: Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
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
                      'Overall Race', 'Day of Week']
for col in categorical_columns:
   df[col] = label_encoder.fit_transform(df[col])

# Convert 'Domestic Violence Incident' column to binary (True/False -> 1/0)
df['Domestic Violence Incident'] = df['Domestic Violence Incident'].astype(int)

# OneHotEncode 'City' dynamically
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Use sparse_output instead of sparse
city_encoded = encoder.fit_transform(df[['City']])
city_encoded_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(['City']))
df = pd.concat([df.reset_index(drop=True), city_encoded_df.reset_index(drop=True)], axis=1)

# Drop the original 'City' column
df.drop(columns=['City'], inplace=True)

# Step 4: Feature Selection
features = ['Victim Age', 'Overall Race', 'Zip Code', 'Hour', 'Day of Week', 'Month'] + list(city_encoded_df.columns)
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

# Step 8: Make Predictions
y_pred = weighted_model.predict(X_test)

# Step 9: Evaluate the Model
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Save the Model and Encoder
joblib.dump((weighted_model, encoder), 'domestic_violence_model_with_encoder.pkl', compress=3)
print("Model and encoder saved as 'domestic_violence_model_with_encoder.pkl'.")

# Step 11: Plot Feature Importances
feature_importances = pd.Series(weighted_model.feature_importances_, index=features)
plt.figure(figsize=(12, 8))
feature_importances.sort_values().plot(kind='barh')
plt.title('Feature Importances in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
