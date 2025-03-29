import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset with error handling
file_path = "car_price.csv"  # Update with the full path if necessary
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Make sure it's in the correct directory.")
    exit()

# Display basic information
print("Dataset Shape:", data.shape)
print(data.info())
print(data.describe())

# Step 2: Exploratory Data Analysis (EDA)
sns.pairplot(data[['selling_price', 'year', 'km_driven', 'engine', 'max_power']])
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

sns.boxplot(x='fuel', y='selling_price', data=data)
plt.title('Price Distribution by Fuel Type')
plt.show()

# Step 3: Data Preprocessing and Feature Engineering
# Convert selling_price into a binary classification problem
y = np.where(data['selling_price'] > data['selling_price'].median(), 1, 0)  # 1: High Price, 0: Low Price

# Handling missing values and data cleaning
data['engine'] = pd.to_numeric(data['engine'].astype(str).str.replace(' CC', '', regex=False), errors='coerce')
data['max_power'] = pd.to_numeric(data['max_power'].astype(str).str.replace(' bhp', '', regex=False), errors='coerce')

# Drop non-numeric and non-relevant columns
X = data.drop(columns=['name', 'selling_price'])

# Fill missing values
X.fillna(X.median(numeric_only=True), inplace=True)

num_features = ['year', 'km_driven', 'engine', 'max_power', 'seats']
cat_features = ['fuel', 'seller_type', 'transmission', 'owner']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# Step 4: Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Price', 'High Price'], yticklabels=['Low Price', 'High Price'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
