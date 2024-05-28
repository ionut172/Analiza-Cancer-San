Logistic Regression on Haberman's Survival Data
This project demonstrates the application of logistic regression on Haberman's survival dataset to predict the survival status of patients who have undergone surgery for breast cancer.

Dataset Information
The dataset contains the following attributes:

Age: Age of patient at time of operation (numerical)
Year of Operation: Patient's year of operation (year - 1900, numerical)
Number of Positive Axillary Nodes: Number of positive axillary nodes detected (numerical)
Survival Status: Survival status (class attribute)
1 = the patient survived 5 years or longer
2 = the patient died within 5 years
Dataset URL: Haberman's Survival Data

Requirements
bash
Copiază codul
numpy
pandas
matplotlib
seaborn
scikit-learn
Code Overview
python
Copiază codul
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data'
df = pd.read_csv(url, header=None)
df.columns = ['age', 'year_operation', 'nr_nodes', 'survival_status']

# Exploratory Data Analysis
plt.scatter(df['nr_nodes'], df['survival_status'])
plt.show()

plt.boxplot(df['nr_nodes'])
plt.show()

# Data Preprocessing
X = df.drop(columns=['survival_status'])
y = df['survival_status']

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model Training and Evaluation
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error_rate = (y_pred != y_test).sum() / y_test.size

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)
y_pred_scaled = model.predict(X_test_scaled)
error_rate_scaled = (y_pred_scaled != y_test).sum() / y_test.size

# Normalization
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()

X_train_normalized = normalizer.fit_transform(X_train)
X_test_normalized = normalizer.transform(X_test)

model.fit(X_train_normalized, y_train)
y_pred_normalized = model.predict(X_test_normalized)
error_rate_normalized = (y_pred_normalized != y_test).sum() / y_test.size

# Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()

X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

model.fit(X_train_minmax, y_train)
y_pred_minmax = model.predict(X_test_minmax)
error_rate_minmax = (y_pred_minmax != y_test).sum() / y_test.size
Results
Initial Model Error Rate: <initial_error_rate>
Error Rate with Standard Scaling: <error_rate_scaled>
Error Rate with Normalization: <error_rate_normalized>
Error Rate with Min-Max Scaling: <error_rate_minmax>
Conclusion
Different preprocessing techniques such as standard scaling, normalization, and min-max scaling can have a significant impact on the performance of a logistic regression model. In this project, we have demonstrated these effects on Haberman's survival dataset.

License
This project is licensed under the MIT License.

