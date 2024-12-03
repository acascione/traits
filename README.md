# **TRAITS**, a **T**ool for **R**evealing **A**ttributes and **I**dentifying **T**oxic and **S**afe profiles in online social networks

This repository contains code to reproduce experiments for the paper Unsupervised and Interpretable Detection of User Personalities in Online Social Networks, submitted for IDA 2025 conference

### Datasets  
For the PANDORA dataset, we redirect to the official paper [PANDORA Talks: Personality and Demographics on Reddit](https://arxiv.org/pdf/2004.04460v3). The dataset is released by the authors under explicit request.

For Reddit users dataset used for case study, we redirect to [Reddit dataset about the Great Ban moderation intervention](https://zenodo.org/records/14034510).

### Code 

Trained Personality Detection models for each traits are available in:

https://huggingface.co/PwNzDust/extraversion_model

https://huggingface.co/PwNzDust/conscientiousness_model

https://huggingface.co/PwNzDust/agreeableness_model

https://huggingface.co/PwNzDust/neuroticism_model_30

https://huggingface.co/PwNzDust/openness_model

Training pipeline specified in ```LongFormer_training.ipynb```

```Evaluate_LongFormer_Classification.ipynb``` presents the code for evaluation of the trained black box models for personality detection

For the toxicity extraction, we refer to [Perspective API](https://perspectiveapi.com/)

``` extract_text_features.ipynb```  presents the code for extracting interpretable information from single texts

```aggregate_users_features.ipynb```  aggregates the info for each user

```Extract_personality_embeds_PIANO.py``` presents the code for extracting users embeddings

```Clustering_process.ipynb``` reports the application fo k-means algorithm for toxic and safe users clustering

```kfold_val_DT_EBM_LGB.py``` reports the k-fold validation process for Decision Tree, Explaianble Boosting Machine and Light Gradient Boosting. It also includes for each model the hyperparameter used for validation

```kfold_val_PT.py``` reports the k-fold validation process for PivotTree both as classifier and as a selector with DT, KNN and Explaianble Boosting Machine. This code includes PT hyperparameters, similar to ```kfold_val_DT_EBM_LGB.py```. For PT, we refer to the implementation in  https://github.com/msetzu/pivottree.

```Clusters_characterization_.ipynb``` specifies the code for visualization for radar charts and bar charts in the paper

### Additional Info
```interpretable_features.txt``` reports the list of interpretable features obtained through the _extract_features_ function with the resources and API described in the paper text.


