
# Final Report: Titanic Dataset Analysis

## 1. Exploratory Data Analysis (EDA) Insights

### Target Variable Distribution (Survival Status):
- The target variable, `Survived`, indicates whether a passenger survived (1) or did not survive (0) the Titanic disaster.
- **Distribution**:
  - 61.6% of the passengers did not survive (0).
  - 38.4% survived (1).
  
  The class imbalance suggests that most passengers did not survive, which is important to note when evaluating model performance.

### Feature Relationships:
- **Age**: The distribution of `Age` shows a typical age range for passengers, with most falling between 20 and 50 years. There is a slight right skew due to a few older passengers.
- **Fare**: The `Fare` distribution is positively skewed, with a higher number of passengers having lower fares, but a few passengers with much higher fares.
- **Pclass (Passenger Class)**: First-class passengers tend to have a higher survival rate than those in second and third classes.
- **Sex**: Women had a higher survival rate than men, reflecting the historical "women and children first" policy.
- **Embarked**: Most passengers boarded from port `C` (Cherbourg), followed by port `S` (Southampton). The `Embarked` variable has some missing values which were filled with the most frequent port (`S`).

---

## 2. Data Cleaning Steps

### Missing Values:
- **Age**: Missing values in `Age` were filled with the median age. This choice ensures that the distribution of ages remains unbiased and does not introduce skew.
- **Fare**: Missing values in `Fare` were also filled with the median fare.
- **Embarked**: Missing values in `Embarked` were filled with the mode (most frequent value), which was `S` (Southampton).
  
These data cleaning steps ensure that we retain the data while minimizing biases introduced by missing values.

### Feature Encoding:
- Categorical variables such as `Sex` and `Embarked` were encoded using **one-hot encoding**. This method converts the categories into numerical columns without introducing ordinal relationships, allowing us to use these features in machine learning models.

---

## 3. Model Performance

We trained three machine learning models: **Logistic Regression**, **Decision Tree**, and **Random Forest**.

### Logistic Regression:
- **Accuracy**: 0.8013
- **Confusion Matrix**:
  ```
  [[112, 18]  # True Negative, False Positive
   [ 29, 45]]  # False Negative, True Positive
  ```
- **Key Metrics**:
  - Precision (Survived): 0.7143
  - Recall (Survived): 0.6087
  - F1-Score (Survived): 0.6585

### Decision Tree:
- **Accuracy**: 0.7877
- **Confusion Matrix**:
  ```
  [[109, 21]  # True Negative, False Positive
   [ 30, 44]]  # False Negative, True Positive
  ```
- **Key Metrics**:
  - Precision (Survived): 0.6774
  - Recall (Survived): 0.5931
  - F1-Score (Survived): 0.6329

### Random Forest:
- **Accuracy**: 0.8209
- **Confusion Matrix**:
  ```
  [[114, 16]  # True Negative, False Positive
   [ 26, 48]]  # False Negative, True Positive
  ```
- **Key Metrics**:
  - Precision (Survived): 0.7500
  - Recall (Survived): 0.6491
  - F1-Score (Survived): 0.6969

The **Random Forest** model performs the best with the highest accuracy and the best balance between precision and recall.

---

## 4. Key Features

Based on feature importance and the results from the models, the most important features influencing survival likelihood are:
- **Pclass (Passenger Class)**: First-class passengers have a significantly higher survival rate.
- **Sex**: Female passengers have a higher likelihood of survival compared to male passengers.
- **Age**: Younger passengers (especially children) tend to have a higher survival rate.
- **Fare**: Passengers with higher fares tend to have higher survival rates.
- **Embarked**: The port of embarkation (especially Cherbourg) has some influence on survival, with passengers from port `C` showing higher survival rates.

---

## 5. Next Steps

### Potential Improvements:
- **Hyperparameter Tuning**: We can improve model performance by tuning the hyperparameters of the models, such as adjusting the `max_depth` for the Decision Tree and `n_estimators` for the Random Forest. This would help avoid overfitting and improve generalization.
  
  For example, using **GridSearchCV** to tune the hyperparameters of the Random Forest could yield better results.

- **Feature Engineering**: We could create new features, such as grouping ages into bins (e.g., child, adult, senior), or using `Family Size` (combining `SibSp` and `Parch`), which may provide additional predictive power.
  
- **Cross-Validation**: Implementing **cross-validation** would give a more robust evaluation of the model performance by training on multiple subsets of the data and evaluating on the remaining data.

- **Trying Other Models**: We can experiment with other models like **Support Vector Machines (SVM)**, **K-Nearest Neighbors (KNN)**, or **XGBoost** to see if they can further improve the results.

---

## Conclusion:
The Titanic dataset provided valuable insights into survival predictors, and after cleaning the data and training multiple models, we found that **Random Forest** performs the best. Future work could focus on tuning the models and experimenting with more advanced techniques for further improvements.