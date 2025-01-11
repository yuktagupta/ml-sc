import pandas as pd
import numpy as np
from scipy import stats

# Load Titanic dataset (replace with your actual file path)
df = pd.read_csv('titanic.csv')  # Adjust file name if necessary

# Display first few rows to understand the structure
print("Initial Data:")
print(df.head())

# 1. Handle Missing Values
# For simplicity, we will fill missing values in numeric columns with the column mean.
# For categorical columns, we will fill missing values with the mode (most frequent value).

# Fill missing values in numeric columns with the mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# Fill missing values in categorical columns with the mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin'] = df['Cabin'].fillna('Unknown')  # Fill Cabin with a default string if missing

# 2. Handle Inconsistent Formatting
# Convert 'Sex' column to lowercase to standardize the text
df['Sex'] = df['Sex'].str.lower()

# Normalize 'Embarked' column by stripping extra spaces and ensuring consistency in uppercase
df['Embarked'] = df['Embarked'].str.strip().str.upper()

# 3. Handle Outliers using Z-score method
# We'll apply outlier detection on the numeric columns: 'Age' and 'Fare'

numeric_columns = ['Age', 'Fare']

# Calculate Z-scores for numeric columns to detect outliers
z_scores = np.abs(stats.zscore(df[numeric_columns]))

# Set a threshold for Z-scores to identify outliers (e.g., Z > 3)
threshold = 3
outliers = (z_scores > threshold)

# Remove outliers by filtering out rows where any numeric column exceeds the threshold
df_no_outliers = df[(z_scores < threshold).all(axis=1)]

# Display cleaned data
print("\nCleaned Data (after handling missing values, inconsistent formatting, and outliers):")
print(df_no_outliers.head())

# Save the cleaned dataset if needed
df_no_outliers.to_csv('cleaned_titanic_data.csv', index=False)
