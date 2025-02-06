import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load the dataset
titanic_df = pd.read_csv("data/train.csv")

#Display the first five rows:
print(titanic_df.head()) #.head() method displays the n rows from csv file. Default arg = 5

#check the shape (rows, columns)
print("Shape: ", titanic_df.shape) #prints the shape of the dataset (rows and columns)

#Summary statistics for numerical columns
print(titanic_df.describe()) #prints the descriptive statistics (count, mean, median, max, min, etc.) of numerical columns

#Data types and missing values
print(titanic_df.info())


# titanic_df.info(): This method provides a concise summary of the DataFrame, including:
# 
# The index dtype (usually int64 for a standard index).
# The data types of each column (e.g., int64, float64, object (for strings), bool).
# The number of non-null values in each column. This is crucial for identifying missing data.
# The memory usage of the DataFrame.

#find the missing values
#.isnull() determines if the numerical value is missing or not (True or 1 for missing values, False or 0 for non-null values)
#.sum() adds the number of True or 1s (i.e. counts the number of missing values.
print(titanic_df.isnull().sum()) 

#Fill the missing value of age with median age
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace = True)
#.median() method finds the median of 'Age'.
#.fillna() method fills the missing values in 'Age' column with the median age
#inplace = True modifies the original DataFrame instead of making a new DataFrame in pandas.

#Drop data from Cabin because of too many missing value
titanic_df.drop('Cabin', axis = 1, inplace = True)
#.drop() method drops a row (axis = 0) or a column (axis = 1). This method takes the column/row name (string) as an argument.

#Drop two rows of "Embarked" (Missing values)
titanic_df.dropna(subset = ['Embarked'], inplace = True)

# Group by Sex and calculate survival rate
survival_by_sex = titanic_df.groupby('Sex')['Survived'].mean().reset_index()

#Plot for sex vs. survival chances
plt.figure(figsize = (5,4)) #size of the plot
sns.barplot(x = 'Sex', y = 'Survived', data = survival_by_sex) #barplot using seaborn library
plt.title('Survival Rate by Gender') #title of the plot
plt.show() #displays the plot

# Women survived at much higher rate than men.

# Group by class and calculate survival rate
survival_by_class = titanic_df.groupby('Pclass')['Survived'].mean().reset_index()

#Plot for class vs. survival rate
plt.figure(figsize = (5,4))
sns.barplot(x = 'Pclass', y = 'Survived', data = survival_by_class)
plt.title('Survival Rate by Class')
plt.show()

# Passengers from the first class had the highest survival rate

#Plot for age distribution vs. survival rate
plt.figure(figsize = (8,4))
sns.histplot(data = titanic_df, x = 'Age', hue = 'Survived', bins = 30, kde = True)
plt.title('Age Distribution by Survival')
plt.show()

# People in mid to late 20s had highest survival rate.

# Correlation Heatmap 

#converting categorical columns into numerical
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
#drop columns with non-numerical data
cols_to_drop = ['Name', 'Ticket', 'Embarked']
titanic_df_clean = titanic_df.drop(columns = cols_to_drop, errors = 'ignore')

#Compute correlations
corr = titanic_df_clean.corr()

#plot heatmap
plt.figure(figsize = (10,5))
sns.heatmap(corr, annot = True, cmap = 'coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#Plot for fare vs. survival rate
bins = [0, 50, 100, 200, 300, 400, 500, 600]  # bins for the range of fares
labels = ['0-50', '50-100', '100-200', '200-300', '300-400', '400-500', '500+'] #labels for the bins
titanic_df['Fare_Range'] = pd.cut(titanic_df['Fare'], bins=bins, labels=labels, right=False) # right=False makes the bins include the left edge
#pd.cut() method is used to discretize continuous data (like 'Fare') into discrete intervals (bins).
# Calculate survival rate for each fare range
survival_by_fare = titanic_df.groupby('Fare_Range')['Survived'].mean()

print(survival_by_fare)

# Plotting
import matplotlib.pyplot as plt
survival_by_fare.plot(kind='bar')
plt.xlabel('Fare Range')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Fare Range')
plt.show()

# The survival rate is highest for passengers who had paid high fares for the trip. The number of passengers who had paid above 500 had the highest survival rate.

#survival rate and family
survival_by_family = titanic_df.groupby('SibSp')['Survived'].mean().reset_index()
sns.barplot(x='SibSp', y='Survived', data=survival_by_family)
plt.title('Survival rate and SibSp')
plt.show()

# - Women and children were prioritized for lifeboats. 
# - Wealthier passengers (1st class) had better survival rates.
# - Age alone wasn’t the strongest predictor of survival.
# - Passengers who paid higher fare had a higher chance of survival.
# - Passengers with more family members were less likely to survive.

# Load data
titanic_df = pd.read_csv("data/train.csv")

# Drop irrelevant columns
titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

# Separate features (X) and target (y)
X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']

# Define categorical and numerical columns
categorical_features = ['Sex', 'Embarked']
numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

#split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a pipeline
logreg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train the model
logreg_pipeline.fit(X_train, y_train)

# Predict on test data
y_pred_logreg = logreg_pipeline.predict(X_test)

# Create a pipeline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
rf_pipeline.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_pipeline.predict(X_test)

#logistic regression metrics
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg):.2f}")
print(classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))

#random forest metrics
print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

#hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 5]
}
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

#feature importance
import matplotlib.pyplot as plt

feature_names = numerical_features + list(
    rf_pipeline.named_steps['preprocessor']
    .named_transformers_['cat']
    .get_feature_names_out(categorical_features)
)

plt.barh(feature_names, rf_pipeline.named_steps['classifier'].feature_importances_)
plt.title("Feature Importance (Random Forest)")
plt.show()
