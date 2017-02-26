
import numpy
import pandas

from sklearn import model_selection


DATA = pandas.read_csv('../data/sample.csv')

LINE_LENGTH = 50

SEED = 1
PROPORTION = 0.3
NUMBER_OF_SPLITS = 5
DEPENDENT = 'IsClick'
INDEPENDENTS = ['Position', 'HistCTR']

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = model_selection.train_test_split(
    DATA[INDEPENDENTS],
    DATA[DEPENDENT],
    random_state=SEED,
    test_size=PROPORTION
)

STRATIFIED_K_FOLD = model_selection.StratifiedKFold(
    n_splits=NUMBER_OF_SPLITS,
    random_state=SEED,
    shuffle=True
)

CROSS_VALIDATION_GRID = {
    'C': [x/10.0 for x in range(1, 11)],
    # 'class_weight': [None, 'balanced', {1.0: 0.9, 0.0: 0.1}],
    # 'class_weight': ['balanced', {1.0: 0.9, 0.0: 0.1}],
    # 'class_weight': ['balanced'],
    'penalty': ['l1', 'l2']
    # 'penalty': ['l2']
}
