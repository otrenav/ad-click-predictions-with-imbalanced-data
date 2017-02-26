
import numpy

from sklearn import metrics
from sklearn import model_selection

import setup

def print_cross_validation_results(number_of_splits,
                                   cross_validation,
                                   X_train,
                                   X_test,
                                   y_train,
                                   y_test):
    predictions = cross_validation.predict(X_test)
    probabilities = cross_validation.predict_proba(X_test)[:, 1]
    print '\n' + '-' * setup.LINE_LENGTH
    print '- Splits:           ', number_of_splits
    print '- Train score:      ', cross_validation.score(X_train, y_train)
    print '- Test  score:      ', cross_validation.score(X_test, y_test)
    print '- Log loss:         ', metrics.log_loss(y_test, probabilities)
    print '- ROC AUC:          ', metrics.roc_auc_score(y_test, probabilities)
    print '- Best parameters:  ', cross_validation.best_params_
    print '\n- Confusion matrix: \n\n', metrics.confusion_matrix(y_test, predictions), '\n'
    print '\n- Classification report: \n\n', metrics.classification_report(y_test, predictions), '\n'
    print '-' * setup.LINE_LENGTH
