#
# Logistic Regression with Simple Sample
#
# Omar Trejo
# January, 2017
#

from sklearn import linear_model
from sklearn import model_selection

import functions
import setup

model = linear_model.LogisticRegression(class_weight='balanced')

cross_validation = model_selection.GridSearchCV(
    param_grid=setup.CROSS_VALIDATION_GRID,
    cv=setup.STRATIFIED_K_FOLD,
    estimator=model,
    n_jobs=8
)

cross_validation.fit(setup.X_TRAIN, setup.Y_TRAIN)

functions.print_cross_validation_results(
    setup.NUMBER_OF_SPLITS,
    cross_validation,
    setup.X_TRAIN,
    setup.X_TEST,
    setup.Y_TRAIN,
    setup.Y_TEST
)
