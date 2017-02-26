
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split

df = pd.read_csv('sample2.csv')
labels = df['IsClick']


k = 5
seed = 2017
model = LogisticRegression()
scores = 0

for i in range(k):
    X_train, X_test, y_train, y_test = train_test_split(df[['Position','HistCTR']], df['IsClick'], test_size=0.2,stratify=labels, random_state=i*seed)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:,1]
    
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    scores += roc_auc

print "Mean AUC: %f" % (scores/k)
