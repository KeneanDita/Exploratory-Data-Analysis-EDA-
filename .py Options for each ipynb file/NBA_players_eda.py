import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = 'CSV_files/nba_players.csv'  # Make sure this file exists in the same directory
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Display basic information
print(data.head())
print("Dataset Shape:", data.shape)
print(data.info())
print(data.describe(include='all'))

# Step 2: Data Cleaning
# Drop duplicate rows (if any)
data.drop_duplicates(inplace=True)

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Fill missing values if necessary
# Example: Filling missing positions with 'Unknown'
if 'position' in data.columns:
    data['position'].fillna('Unknown', inplace=True)

# Step 3: Exploratory Data Analysis (EDA)
# Count of players per position
plt.figure(figsize=(10, 5))
sns.countplot(x='position', data=data, order=data['position'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Count of Players by Position')
plt.show()

# Count of teams in each conference
plt.figure(figsize=(10, 5))
sns.countplot(x='conference', data=data, order=data['conference'].value_counts().index)
plt.title('Number of Teams by Conference')
plt.show()

# Count of teams in each division
plt.figure(figsize=(12, 6))
sns.countplot(x='division', data=data, order=data['division'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Number of Teams by Division')
plt.show()

# Step 4: Data Processing for Further Analysis
# Create a new feature for full player names if not already present
if 'full_name' not in data.columns:
    data['full_name'] = data['first_name'] + ' ' + data['last_name']

# Step 5: Summary Statistics
print("\nSummary Statistics by Conference:")
print(data.groupby('conference').size())

print("\nSummary Statistics by Position:")
print(data.groupby('position').size())

print("\nSummary Statistics by Division:")
print(data.groupby('division').size())

# Step 6: Save Cleaned Data
data.to_csv('nba_players_cleaned.csv', index=False)
print("\nCleaned dataset saved as 'nba_players_cleaned.csv'")
