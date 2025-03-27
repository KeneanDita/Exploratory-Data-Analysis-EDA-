import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("CSV_files/heart_disease.csv")

data.shape

print(data['target'].value_counts())
data.isnull().sum()
data.describe()

data.value_counts('thal')

sns.histplot(data['thalach'])
plt.title("Maximum Heart Rate Achieved")
# plt.savefig('Maximum_Heart_Rate_Achieved_Distribution.png', dpi=300)
plt.show()

sns.boxplot(x=data['age'])
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data = pd.get_dummies(data, columns=['cp', 'thal'], drop_first=True)
scaler = StandardScaler()
data[['age', 'chol', 'thalach']] = scaler.fit_transform(data[['age', 'chol', 'thalach']])

columns_to_drop = ["age_group", "cp_1", "cp_2", "cp_3", "thal_1", "thal_2", "thal_3"]
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

X = data.drop(columns=["target"])
y = data["target"]
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred_tree))

export_graphviz(tree, out_file='decision_tree.dot', 
                feature_names=X.columns, 
                class_names=sorted(y.unique().astype(str)), 
                label='all', 
                rounded=True, filled=True)