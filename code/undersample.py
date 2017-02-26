#
# Logistic Regression with Random Undersample
#
# Omar Trejo
# January, 2017
#

from imblearn import under_sampling
from collections import Counter
from sklearn import linear_model
from sklearn import model_selection

import functions
import setup

random_undersampler = under_sampling.RandomUnderSampler(
    random_state=setup.SEED,
    ratio='auto'
)

X_train_resampled, y_train_resampled = random_undersampler.fit_sample(
    setup.X_TRAIN,
    setup.Y_TRAIN
)

print "-" * setup.LINE_LENGTH
print "Before resample: {}".format(Counter(setup.Y_TRAIN))
print "After resample:  {}".format(Counter(y_train_resampled))
print "-" * setup.LINE_LENGTH

model = linear_model.LogisticRegression(class_weight='balanced')

cross_validation = model_selection.GridSearchCV(
    param_grid=setup.CROSS_VALIDATION_GRID,
    cv=setup.STRATIFIED_K_FOLD,
    estimator=model,
    n_jobs=8
)

cross_validation.fit(X_train_resampled, y_train_resampled)

functions.print_cross_validation_results(
    setup.NUMBER_OF_SPLITS,
    cross_validation,
    X_train_resampled,
    setup.X_TEST,
    y_train_resampled,
    setup.Y_TEST
)
