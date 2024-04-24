'''Code to fit and evaluate logit models for checking corectness of the segmentation. 
Include features you want to fit the logit on into the FEATURES_TO_EVALUATE list.
Input: A dictionary in json format containing values of intersection, Levenshtein and text presence, as well as the information whether the segmentation was correct or incorrect.
Output: Evaluated logit models with their statistical evaluation revealing, how good these perfom for the task of determining, whether the segmentation was correct or incorrect.'''

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

import pandas as pd
import numpy as np
import json
import joblib

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

FEATURES_TO_EVALUATE = ['Hausdorff', 'BordersText'] # set the features or their combinations from the following list: ['Intersection', 'IoU', 'Hausdorff', 'Levenshtein', 'BordersText']

# 1. Loading data
file_path = 'path/to/dictionary.json'

with open(file_path, 'r') as file:
    results_dict = json.load(file)

data = []
for tag, tag_data in results_dict.items():
    for page, page_data in tag_data.items():
        for region, region_data in page_data.items():
            if region_data['CorrectSegmentation'] is not None:
                data.append([tag, page, region, region_data['Intersection'], region_data['IoU'], region_data['Hausdorff'], region_data['Levenshtein'], region_data['BordersText'], region_data['CorrectSegmentation']])

df = pd.DataFrame(data, columns=['tag', 'page', 'region', 'Intersection', 'IoU', 'Hausdorff', 'Levenshtein', 'BordersText', 'CorrectSegmentation'])
df['BordersText'] = df['BordersText'].astype(int)

if 'Levenshtein' in FEATURES_TO_EVALUATE:
    df = df.dropna(subset=['Levenshtein'])

# 2. Ensure balanced classes 
class_labels = df['CorrectSegmentation'].unique()
majority_class_label = max(class_labels, key=lambda x: (df['CorrectSegmentation'] == x).sum())
minority_class_label = min(class_labels, key=lambda x: (df['CorrectSegmentation'] == x).sum())

majority_class = df[df['CorrectSegmentation'] == majority_class_label]
minority_class = df[df['CorrectSegmentation'] == minority_class_label]

undersampled_majority = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)
undersampled_df = pd.concat([undersampled_majority, minority_class])

# 3. Split train and test data
targets = df['CorrectSegmentation']
features = df[FEATURES_TO_EVALUATE]

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=24, stratify=df.CorrectSegmentation)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

logit = sm.Logit(y_train, X_train)

# Fit the model
logit = logit.fit()

# Make predictions on the test set
y_pred = logit.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
score = roc_auc_score(y_test, y_pred)

print("Accuracy:", np.round(accuracy, 3))
print("F1 Score:", np.round(f1, 3))
print(f"ROC AUC: {score:.3f}")
print(logit.summary())

# 5. Save the model
model_file_path =f'path/to/save/model/logit_model_{FEATURES_TO_EVALUATE}.pkl'
joblib.dump(logit, model_file_path)
