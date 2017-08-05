# recommendation system

[![Datmo Model](https://datmo.io/shabazp/recommendation-system/badge.svg)](https://datmo.io/shabazp/recommendation-system) 

Recommendation system using lightfm

In this example, we’ll build an implicit feedback recommender using the Movielens 100k dataset (http://grouplens.org/datasets/movielens/100k/).

The code behind this example is available as a Jupyter notebook

LightFM includes functions for getting and processing this dataset, so obtaining it is quite easy.

This downloads the dataset and automatically pre-processes it into sparse matrices suitable for further calculation. In particular, it prepares the sparse user-item matrices, containing positive entries where a user interacted with a product, and zeros otherwise.

We’re going to use the WARP (Weighted Approximate-Rank Pairwise) model. WARP is an implicit feedback model: all interactions in the training matrix are treated as positive signals, and products that users did not interact with they implicitly do not like. The goal of the model is to score these implicit positives highly while assigining low scores to implicit negatives.

Done! We should now evaluate the model to see how well it’s doing. We’re most interested in how good the ranking produced by the model is. Precision@k is one suitable metric, expressing the percentage of top k (in our case we use 5) items in the ranking the user has actually interacted with. lightfm implements a number of metrics in the evaluation module.
