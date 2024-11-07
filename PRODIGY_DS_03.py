import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
titanic = pd.read_csv('test.csv')  # Replace 'your_file.csv' with your actual file path

# Check for missing values
print(titanic.isnull().sum())

# Fill missing values in 'Age' with the median age
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the mode
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to excessive missing values
titanic.drop(columns=['Cabin'], inplace=True)

# Convert 'Sex' to numeric
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})

# Check for missing values again
print(titanic.isnull().sum())

# Display basic statistics
print(titanic.describe())

# Select only numeric columns for the correlation matrix
numeric_cols = titanic.select_dtypes(include=['float64', 'int64']).columns

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(titanic[numeric_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Plot gender distribution
sns.countplot(x='Sex', data=titanic)
plt.title('Gender Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Plot age distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=titanic, x='Age', kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Plot class distribution
sns.countplot(x='Pclass', data=titanic)
plt.title('Class Distribution')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()

# Plot fare distribution by class
plt.figure(figsize=(12, 6))
sns.boxplot(x='Pclass', y='Fare', data=titanic)
plt.title('Fare Distribution by Class')
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.show()
