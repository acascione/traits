from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import lightgbm as lgb
from sklearn.metrics import make_scorer, accuracy_score
import itertools

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import silhouette_score  

random_state = 42

#changing the dataset implies changing the problem and so resulted reported in the finale performance table in the paper
#here we are using the clustering on embedding and prediction on interpretable featurs


df_clustering_name = f'INTERPRETABLE_xmeans_TOXIC_NON_TOXIC_DIVISION_embeddings_CONCAT_3.csv' #this contains the clustering results
df_classification_name = f'INTERPRETABLE_xmeans_TOXIC_NON_TOXIC_DIVISION_interpretable_CONCAT.csv' #this contains the classification results

df_clustering = pd.read_csv(df_clustering_name)
df_classification = pd.read_csv(df_classification_name)


y = df_classification.cluster_preds #assign to Y the cluster predictions in the dataframe

df_classification = df_classification.drop(columns = ['cluster_preds']) #drop the cluster column from the predictive variable

X = df_classification.values


dict_lables = {k : i for i, k in enumerate(list(set(y)))}
dict_lables

y = np.array([dict_lables[i] for i in y])



# Define hyperparameter grid for search
# This is defined for Decision Tree Classifier
param_grid = {
    'max_depth': [4, 16, 32],             
    'min_samples_split': [2, 5, 10],            
    'min_samples_leaf': [1, 2, 4],            
}

## This is explored for Explainable Boosting Machine
"""
param_grid = {
    'max_leaves': [3,4],                
    'learning_rate': [0.01, 0.05, 0.1], 
    'smoothing_rounds' : [200,500],
    #'n_estimators': [100, 200, 300, 500], 
    'reg_alpha': [0.0, 0.1],          
    'reg_lambda': [0.0, 0.1],         
    'n_jobs' : [4]
}
"""

"""
# Define hyperparameter grid for Logistic Regression
param_grid = {
    'C': [0.001, 0.01, 0.1, 1],  # Regularization strength
    'penalty': ['l2'],  # Type of regularization
    'solver': ['lbfgs'],  # Optimization solver (supports L1 and ElasticNet)
    'max_iter': [1000]  # Number of iterations
}
"""

# Define hyperparameter grid for knn 
"""
param_grid = {
    'n_neighbors': [1, 3, 5],  # Number of neighbors
    'metric': ['euclidean'],   # Use Euclidean distance
    'weights': ['uniform', 'distance']  # Weighting strategy
}
"""






## This is explored for Light Gradient Boosting Machine
"""
param_grid = {
    'num_leaves': [32],                
    'max_depth': [-1, 5, 10],           
    'learning_rate': [0.01, 0.05, 0.1], 
    'n_estimators': [100, 200, 300, 500], 
    'reg_alpha': [0.0, 0.1],          
    'reg_lambda': [0.0, 0.1],           
    'n_jobs' : [4]
}

"""

keys, values = zip(*param_grid.items())
configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]


# Define the classifier you want to use
model = DecisionTreeClassifier()
# model = ExplainableBoostingClassifier() for EBM
# model = lgb.LGBMClassifier()  for LGB
# model =  LogisticRegression()  for Logistic Reg
# model =  KNeighborsClassifier()  for knn

kf = KFold(n_splits=5, shuffle=True, random_state=42)

unique_classes = sorted(set(y))

# Define scoring metrics for each class
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'precision_weighted': make_scorer(precision_score, average='weighted'),
    'recall_weighted': make_scorer(recall_score, average='weighted')
}

# Add class-specific metrics
for cls in unique_classes:
    break
    scoring[f'precision_class_{cls}'] = make_scorer(precision_score, average=None, labels=[cls])
    scoring[f'recall_class_{cls}'] = make_scorer(recall_score, average=None, labels=[cls])
    scoring[f'f1_class_{cls}'] = make_scorer(f1_score, average=None, labels=[cls])

# Set up GridSearchCV with multiple scoring metrics
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=scoring,
    refit='f1_weighted',  # Optimize for f1_weighted, but collect all metrics
    cv=kf,
    n_jobs=4
)

grid_search.fit(X, y)


# Convert the grid search results to a DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

model_name = 'DT'
#model_name = 'EBM'
#model_name 'LGB'

name_task = f'{model_name}_{df_clustering_name}'


# Save the DataFrame to a CSV file (optional)
file_name = f'results/k_fold_crossval_results_{name_task}.csv'

results_df.to_csv(file_name, index=False)


