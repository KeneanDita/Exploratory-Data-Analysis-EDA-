# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset
data = pd.read_csv('CSV_files/titanic.csv')

# Check for missing values and summarize the missing data
print(data.isnull().sum())  # Summarize the number of missing values in each column
print(data.value_counts(['Embarked']))  # Check the value counts for 'Embarked'
print(data.value_counts(['Survived']))  # Check the value counts for 'Survived'

# Fill missing values in 'Fare' with the median value
data['Fare'] = data['Fare'].fillna(3.445)

# Drop rows where both 'Age' and 'Cabin' are missing (too many NaNs in 'Cabin')
data = data.dropna(subset=['Age', 'Cabin'], how='all')  # Removes rows where both are NaN

# Drop rows with missing 'Age' values
data = data.dropna(subset=['Age'])

# Provide a summary of the dataset's statistics
data.describe()

# Visualize the 'Age' distribution using a histogram
sns.histplot(data['Age'])
plt.title("Age Distribution")
plt.savefig('Age Distribution.png', dpi=300)
plt.show()

# Data cleaning and feature engineering
# Drop duplicates based on the 'Name' column, keeping the first occurrence
data = data.drop_duplicates(subset=['Name'], keep='first')
print(f"Number of duplicated rows: {data.duplicated().sum()}")  # Check the number of duplicates after dropping

# Create a new feature 'FamilySize' by adding 'SibSp' and 'Parch' and adding 1 for the individual
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Extract 'Title' from the 'Name' column using a regular expression
data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Label Encoding: Convert 'Pclass' into numeric values (0, 1, 2)
label_encoder = LabelEncoder()
data['Pclass'] = label_encoder.fit_transform(data['Pclass'])

# Label Encoding: Convert 'Sex' into numeric values (1 for male, 0 for female)
data['Sex'] = label_encoder.fit_transform(data['Sex'])

# Standardizing 'Age' and 'Fare' using StandardScaler
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Drop irrelevant columns from the dataset before modeling
data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Fill missing values in 'Age', 'Fare', and 'Embarked' columns
df = data.copy()  # Use a separate copy for modeling
df["Age"] = df["Age"].fillna(df["Age"].median())  # Fill missing 'Age' with the median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())  # Fill missing 'Fare' with the median
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # Fill missing 'Embarked' with the mode (most frequent value)

# Encode categorical variables: 'Sex' and 'Embarked'
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Define features (X) and target (y) for machine learning
X = df.drop(columns=["Survived"])  # Features (excluding the target variable 'Survived')
y = df["Survived"]  # Target variable 'Survived'

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Decision Tree Classifier
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tree = tree.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred_tree))

# Export the trained Decision Tree to a .dot file for visualization
export_graphviz(tree, out_file='decision_tree.dot', 
                feature_names=X.columns, 
                class_names=sorted(y.unique().astype(str)), 
                label='all', 
                rounded=True, filled=True)

data.to_csv('titanic_cleaned.csv', index=False)
print("\nCleaned dataset saved as 'titanic_cleaned.csv'")
