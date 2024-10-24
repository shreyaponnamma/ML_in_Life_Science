import pandas as pd
import requests
import re
from itertools import product
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



#url of data
url = 'https://raw.githubusercontent.com/BackofenLab/MLLS-exercise-SS22/main/week_2/ELAVL1_PARCLIP'

#Download data
response = requests.get(url)
data = response.text
data_lines = data.splitlines()

#display data
# print(data)

ids = []
sequences = []
targets = []

#parse data
for i in range(0, len(data_lines), 2):
    id_target = data_lines[i]
    sequence = data_lines[i+1]

        # Use regex to extract the ID and target from the first line
    match = re.match(r'^>(ID\d+)\|(\d)$', id_target)
    if match:
        sample_id = match.group(1)
        target = int(match.group(2))
        
        # Append the extracted information to respective lists
        ids.append(sample_id)
        targets.append(target)
        sequences.append(sequence)

# Create a DataFrame to store the processed data
df = pd.DataFrame({'ID': ids, 'Target': targets, 'Sequence': sequences})

# Display the first few rows of the DataFrame
# print(df.head())

# Step 3: Generate k-mer features (k=3)
bases = ['A', 'U', 'C', 'G']
k_mers = [''.join(p) for p in product(bases, repeat=3)]

def k_mer_featurize(sequence, k=3):
    k_mer_counts = Counter([sequence[i:i+k] for i in range(len(sequence) - k + 1)])
    return [k_mer_counts[k_mer] for k_mer in k_mers]

# Apply the featurization to each sequence
k_mer_features = [k_mer_featurize(seq) for seq in df['Sequence']]

# Create a DataFrame for the k-mer features
k_mer_df = pd.DataFrame(k_mer_features, columns=k_mers)

# Combine the k-mer features with the original target column
processed_df = pd.concat([df[['ID', 'Target']], k_mer_df], axis=1)

# Display the first few rows of the processed DataFrame
print(processed_df.head())

# Split the dataset into features (3-mers) and target
X = processed_df.drop(columns=['ID', 'Target']) 
y = processed_df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()

# Train Logistic Regression
log_reg.fit(X_train, y_train)

# Train Random Forest
rf.fit(X_train, y_train)

# Make predictions with both models
y_pred_log_reg = log_reg.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Predict probabilities for AUROC calculation
y_prob_log_reg = log_reg.predict_proba(X_test)[:, 1]
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Calculate metrics for Logistic Regression
metrics_log_reg = {
    'Accuracy': accuracy_score(y_test, y_pred_log_reg),
    'Precision': precision_score(y_test, y_pred_log_reg),
    'Recall': recall_score(y_test, y_pred_log_reg),
    'F1 Score': f1_score(y_test, y_pred_log_reg),
    'AUROC': roc_auc_score(y_test, y_prob_log_reg)
}

# Calculate metrics for Random Forest
metrics_rf = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1 Score': f1_score(y_test, y_pred_rf),
    'AUROC': roc_auc_score(y_test, y_prob_rf)
}

print(metrics_log_reg, metrics_rf)
