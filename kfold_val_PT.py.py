from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import lightgbm as lgb
from sklearn.metrics import make_scorer, accuracy_score
import itertools

from RuleTree_stz import *
from PivotTree_stz import *

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, KFold


import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import silhouette_score  

random_state = 42


def extract_values_parallel(data):
    result = []

    for item in data:
        node_id = int(item[0].split(' ')[1])  # Extracting the node ID as an integer

        if 'pivot:' in item[0]:
            value = int(item[0].split('pivot: ')[-1])  # Extracting the pivot value as an integer
            result.append({'node_id': node_id, 'value': value, 'type': 'pivot'})

        elif 'medoid:' in item[0]:
            value = int(item[0].split('medoid: ')[-1])  # Extracting the medoid value as an integer
            result.append({'node_id': node_id, 'value': value, 'type': 'medoid'})

    return result


df_clustering_name = f'INTERPRETABLE_xmeans_TOXIC_NON_TOXIC_DIVISION_embeddings_CONCAT_3.csv' #this contains the clustering results
df_classification_name = f'INTERPRETABLE_xmeans_TOXIC_NON_TOXIC_DIVISION_interpretable_CONCAT.csv' #this contains the classification results

y = df_classification.cluster_preds #assign to Y the cluster predictions in the dataframe

df_classification = df_classification.drop(columns = ['cluster_preds']) #drop the cluster column from the predictive variable

X = df_classification.values


dict_lables = {k : i for i, k in enumerate(list(set(y)))}
dict_lables

y = np.array([dict_lables[i] for i in y])


from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import numpy as np
import pandas as pd

# Initialize cross-validation and results storage
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = defaultdict(list)

# Define depths and configurations for PivotTree
depths = [2, 3, 4, 5, 6, 7, 8]

# Define classifiers for PivotTree selector
dt4 = DecisionTreeClassifier(max_depth=4, random_state=42)
dt6 = DecisionTreeClassifier(max_depth=6, random_state=42)
knn5 = KNeighborsClassifier(n_neighbors=5)
ebm = ExplainableBoostingClassifier(random_state=42)
classifiers = {'dt4': dt4, 'dt6': dt6, 'knn5': knn5, 'ebm': ebm, 'None' : None}

scoring = {
    'accuracy': accuracy_score,
    'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
    'precision_weighted': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
    'recall_weighted': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted')
}

# Add class-specific metrics
unique_classes = np.unique(y)
for cls in unique_classes:
    scoring[f'precision_class_{cls}'] = lambda y_true, y_pred, cls=cls: precision_score(y_true, y_pred, labels=[cls], average=None)[0]
    scoring[f'recall_class_{cls}'] = lambda y_true, y_pred, cls=cls: recall_score(y_true, y_pred, labels=[cls], average=None)[0]
    scoring[f'f1_class_{cls}'] = lambda y_true, y_pred, cls=cls: f1_score(y_true, y_pred, labels=[cls], average=None)[0]


metric = 'euclidean'
# Iterate over depths and base stump configurations
for depth in depths:
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            print(f"Fold {fold_idx + 1} - Depth {depth} - Config {config_name}")
            
            # Split data into training and test sets
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Generate distance matrix for training set
            distance_matrix = pairwise_distances(X_train, metric='euclidean')

            # Initialize RuleTreeClassifier with current depth and base stump configuration
            pivottree = PivotTree(random_state = random_state,
                      max_depth = max_depth_clf, distance_matrix = distance_matrix,
                      prune_tree = True, pairwise_metric = metric)

            # Fit the model and predict
            pivottree.fit(X_train, y_train)
            y_pred = pivottree.predict(X_test)

            # Calculate pivot types
            data = extract_values_parallel(pivottree)
            all_discs = np.concatenate(data['discs']).tolist()  
            all_descs = np.concatenate(data['descs']).tolist()
            all_cands = np.concatenate(data['cands']).tolist()
            
            # Handling 'used' where it might contain tuples or strings
            all_used = []
            for item in data['used']:
                if isinstance(item, tuple):
                    all_used.extend(item)  # If it's a tuple, flatten and add elements
                else:
                    all_used.append(item)  # If it's a string (or anything else), just append

            # Organize pivot types
            types_of_pivots = {'disc': all_discs, 'desc': all_descs, 'cands': all_cands, 'used': all_used}

            # Evaluate each classifier on each pivot type
            for pivot_type_name, list_pivot in types_of_pivots.items():
                for classifier_name, classifier in classifiers.items():
                    if classifier is None or classifier_name == 'None':
                        y_pred_pivot = y_pred
                    else:
                        classifier.fit(pairwise_distances(X_train, X_train[np.array(list_pivot)]), y_train)
                        y_pred_pivot = classifier.predict(pairwise_distances(X_test, X_train[np.array(list_pivot)]))
                        
                    # Calculate and store metrics for each classifier and pivot type
                    fold_results = defaultdict(list)
                    for metric_name, metric_func in scoring.items():
                        score = metric_func(y_test, y_pred_pivot)
                        fold_results[metric_name].append(score)

                    # Average fold results for each metric
                    averaged_results = {metric: np.mean(scores) for metric, scores in fold_results.items()}

                    # Store results for the current configuration, classifier, and pivot type
                    config_results = {
                        'depth': depth,
                        'base_stump': config_name,
                        'pivot_type': pivot_type_name,
                        'classifier': classifier_name,
                        **averaged_results
                    }
                    for key, value in config_results.items():
                        results[key].append(value)

# Convert results to DataFrame for easy analysis and save to CSV
results_df = pd.DataFrame(results)

model_name = 'PT'


name_task = f'{model_name}_{df_clustering_name}'


# Save the DataFrame to a CSV file (optional)
file_name = f'results/k_fold_crossval_results_{name_task}.csv'

results_df.to_csv(file_name, index=False)

