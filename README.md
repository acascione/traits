# **TRAITS**, a **T**ool for **R**evealing **A**ttributes and **I**dentifying **T**oxic and **S**afe profiles in online social networks

This repository contains code to reproduce experiments for the paper Unsupervised and Interpretable Detection of User Personalities in Online Social Networks, submitted for IDA 2025 conference

### Datasets  
For the PANDORA dataset, we redirect to the official paper [PANDORA Talks: Personality and Demographics on Reddit](https://arxiv.org/pdf/2004.04460v3). The dataset is released by the authors under explicit request.

For Reddit users dataset used for case study, we redirect to [Reddit dataset about the Great Ban moderation intervention](https://zenodo.org/records/14034510).

### Code 

TODO

```kfold_val_DT_EBM_LGB.py``` reports the k-fold validation process for Decision Tree, Explaianble Boosting Machine and Light Gradient Boosting. It also inclue for each model the hyperparameter used for validation

```kfold_val_PT.py``` reports the k-fold validation process for PivotTree both as classifier and as a selector with DT, KNN and Explaianble Boosting Machine. We refer to the implementation in  https://github.com/msetzu/pivottree - this code includes PT hyperparameters, as above


### Additional Info
```interpretable_features.txt``` reports the list of interpretable features obtained through the _extract_features_ function with the resources and API described in the paper text.


