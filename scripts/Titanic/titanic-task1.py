# %%
import pandas as pd

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
# merge
df = pd.concat([train_data, test_data], axis = 0)
df.rename(columns={'y':'target'},inplace=True)
df

# %%
def dataset_info(df):
    # Check if the input is a pandas DataFrame
    if isinstance(df, pd.DataFrame):
        # Count the total number of missing values in the DataFrame
        is_na = df.isna().sum().sum()
        # Print information about the DataFrame
        print("Datatype shape =", df.shape)
        print("Any null values =", is_na, "\n")
        
        # Get column names, data types, and unique values for each column
        col = df.columns
        datatype = df.dtypes
        uniq = df.nunique()
        
        # Print column-wise information
        print("\033[1m", "S.NO ", " Column", "  Datatype", "  Unique Data", "\n")
        for i in range(len(df.columns)):
            print("%d %10s %10s %10s" % (i + 1, col[i], datatype[i], uniq[i]))

# %%
dataset_info(df)

# %%
# Identify and categorize columns with categorical data
catagorical_data = [j for j in df.columns if df[j].dtype == "O"]
print("Keys with categorical dataset are:", "\033[1m", catagorical_data)

# Identify and categorize columns with numerical data
num = [k for k in df.columns if df[k].dtype != "O"]
print("\033[0m", "Keys with numerical dataset are:", "\033[1m", num)

# %%
import sklearn.datasets
from sklearn.model_selection import train_test_split

# Mapping 'no' to 0 and 'yes' to 1 in the 'target' column
df['target'] = df['target'].map({'no': 0, 'yes': 1})

# Separating features (X) and target variable (y)
x = df.drop(['target'], axis=1)
y = df['target']

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

X_train_original = X_train
X_test_original = X_test

# Printing the shapes of the training and testing sets
print("Elements in X_train:", X_train.shape)
print("Elements in X_test:", X_test.shape)
print("Elements in Y_train:", Y_train.shape)
print("Elements in Y_test:", Y_test.shape)

# %%
X_train

# %%
X_test

# %%
import category_encoders as ce

# Specify categorical columns for encoding
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Initialize OrdinalEncoder
encoder = ce.OrdinalEncoder(cols=categorical_columns)

# Fit and transform the training set
X_train = encoder.fit_transform(X_train)

# Transform the test set using the encoder fitted on the training set
X_test = encoder.fit_transform(X_test)

# Display the column names in the encoded training set
X_train.keys
X_test.head()

# %% [markdown]
# # Random Forest Algorithm

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Instantiate RandomForestClassifier
RF = RandomForestClassifier()

# Fit the model on the training set
RF.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = RF.predict(X_test)

# Calculate accuracy score
Accuracy = accuracy_score(Y_test, Y_pred)

# Calculate confusion matrix
conf = confusion_matrix(Y_test, Y_pred)

# Generate classification report
report = classification_report(Y_test, Y_pred)

# Print the results
print(f"Accuracy score of Random Forest Algorithm is {Accuracy * 100:.2f}%")
print(f"Confusion Matrix:\n{conf}\n")
print(f"Classification Report:\n{report}")

# %%


# %% [markdown]
# remove less contributing attribute

# %%
f_score=pd.Series(RF.feature_importances_,index=X_train.columns).sort_values(ascending=True)
f_score

# %%
import matplotlib.pyplot as plt

# Plotting feature importances
f_score.plot(kind='barh', figsize=(10, 8))
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Classifier - Feature Importance')
plt.show()


# %% [markdown]
# remove outlier for scaling

# %%
from sklearn.preprocessing import StandardScaler

# Remove 'default' and 'loan' columns from training and testing sets
X_train = X_train.drop(columns=['default', 'loan'])
X_test = X_test.drop(columns=['default', 'loan'])

# Instantiate StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training set and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate RandomForestClassifier with 1000 estimators
RF = RandomForestClassifier(n_estimators=1000)

# Fit the model on the modified and scaled training set
RF.fit(X_train_scaled, Y_train)

# Make predictions on the modified and scaled test set
Y_pred = RF.predict(X_test_scaled)

# Calculate accuracy score, confusion matrix, and classification report
Accuracy = accuracy_score(Y_test, Y_pred)
conf = confusion_matrix(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

# Print the results
print(f"Accuracy score of Random Forest Algorithm is {Accuracy * 100:.2f}%")
print(f"Confusion Matrix:\n{conf}\n")
print(f"Classification Report:\n{report}")

# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get predicted probabilities for the positive class
y_pred_proba = RF.predict_proba(X_test_scaled)[:, 1]

# Calculate false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_true=Y_test, y_score=y_pred_proba)

# Calculate ROC-AUC score
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# %%
y_pred_proba

# %%
import numpy as np

# Assuming y_pred_proba is a NumPy array or can be converted to one
y_pred_proba = 1 - np.array(y_pred_proba)

# Round the values to 3 decimals
y_pred_proba = np.round(y_pred_proba, 3)

# %%
y_pred_proba

# %%
# Create a DataFrame with 'id' and 'y' columns
result_df = pd.DataFrame({'id': X_test['id'], 'y': y_pred_proba})

# Sort the DataFrame by ascending 'id'
result_df = result_df.sort_values(by='id', ascending=True)

# Save the DataFrame to a CSV file
result_df.to_csv('predictions_randomforest.csv', index=False, header=['id', 'y'])

# %% [markdown]
# # KNN Classifier

# %%
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train,Y_train)

Y_predicted=KNN.predict(X_test)

print(f"accuracy of the model is {accuracy_score(Y_test,Y_predicted)*100} ")

# %% [markdown]
# # Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(X_train,Y_train)

Y_predicted=log.predict(X_test)

print(f"accuracy of the model is {accuracy_score(Y_test,Y_predicted)*100} ")

# %% [markdown]
# # Support Vector Machine (SVM)

# %%
from sklearn.svm import SVC

# Instantiate SVM classifier
svm_model = SVC(probability=True)

# Fit the model on the training set
svm_model.fit(X_train_scaled, Y_train)

# Make predictions on the test set
Y_predicted_svm = svm_model.predict(X_test)

# Calculate accuracy score
accuracy_svm = accuracy_score(Y_test, Y_predicted_svm)

# Print the accuracy
print(f"Accuracy of the SVM model is {accuracy_svm * 100:.2f}%")


# %%
# Y_predicted_svm_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

# # Assuming y_pred_proba is a NumPy array or can be converted to one
# Y_predicted_svm_proba = 1 - np.array(Y_predicted_svm_proba)

# # Round the values to 3 decimals
# Y_predicted_svm_proba = np.round(y_pred_proba, 3)

# %%
# # Create a DataFrame with 'id' and 'y' columns
# result2_df = pd.DataFrame({'id': X_test['id'], 'y': Y_predicted_svm_proba})

# # Sort the DataFrame by ascending 'id'
# result2_df = result2_df.sort_values(by='id', ascending=True)

# # Save the DataFrame to a CSV file
# result2_df.to_csv('predictions_svm.csv', index=False, header=['id', 'y'])

# %% [markdown]
# # XGBoots

# %%
import xgboost as xgb

# Instantiate XGBoost classifier
xgb_model = xgb.XGBClassifier()

# Fit the model on the training set
xgb_model.fit(X_train, Y_train)

# Make predictions on the test set
Y_predicted_xgb = xgb_model.predict(X_test)

# Calculate accuracy score
accuracy_xgb = accuracy_score(Y_test, Y_predicted_xgb)

# Print the accuracy
print(f"Accuracy of the XGBoost model is {accuracy_xgb * 100:.2f}%")


# %% [markdown]
# # Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Instantiate Decision Tree classifier
dt_model = DecisionTreeClassifier()

# Fit the model on the training set
dt_model.fit(X_train, Y_train)

# Make predictions on the test set
Y_predicted_dt = dt_model.predict(X_test)

# Calculate accuracy score
accuracy_dt = accuracy_score(Y_test, Y_predicted_dt)

# Print the accuracy
print(f"Accuracy of the Decision Tree model is {accuracy_dt * 100:.2f}%")

# Optionally, you can print other evaluation metrics like confusion matrix and classification report
conf_dt = confusion_matrix(Y_test, Y_predicted_dt)
report_dt = classification_report(Y_test, Y_predicted_dt)

print(f"Confusion Matrix:\n{conf_dt}\n")
print(f"Classification Report:\n{report_dt}")



