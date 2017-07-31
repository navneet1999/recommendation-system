import numpy as np
import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import cPickle
import os
import json
import pickle
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
%matplotlib inline
import random
random.seed(0)
# Force matplotlib to not use any Xwindows backend.

# Load the MovieLens 100k dataset. Only five
# star ratings are treated as positive.
data = fetch_movielens(min_rating=4.0)

# Instantiate and train the model
model = LightFM(loss='warp')
model.fit(data['train'], epochs=40, num_threads=2)

# Evaluate the trained model
train_precision = precision_at_k(model, data['train'], k=5).mean()
test_precision = precision_at_k(model, data['test'], k=5).mean()
print test_precision

# save the classifier
stats = {"train_precision": str(train_precision),"test_precision":str(test_precision)}
print stats
model_filename = os.path.join(os.environ['OUTPUT_DIR'],'model.dat')
pickle.dump(model, open(model_filename, 'wb'))
stats_filename = os.path.join(os.environ['OUTPUT_DIR'],'stats.json')
with open(stats_filename, 'wb') as f:
    f.write(json.dumps(stats))
    
def sample_recommendation(model, data, user_ids):


    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)

sample_recommendation(model, data, [3, 25, 450])    