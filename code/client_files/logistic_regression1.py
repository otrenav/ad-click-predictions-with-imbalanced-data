
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split


df = pd.read_csv('sample2.csv')

X_train, X_test, y_train, y_test = train_test_split(df[['Position','HistCTR']], df['IsClick'], test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)


pred = model.predict_proba(X_test)

print 'log_loss :' , log_loss(y_test,pred)
