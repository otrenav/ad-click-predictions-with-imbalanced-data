
| [Website](http://links.otrenav.com/website) | [Twitter](http://links.otrenav.com/twitter) | [LinkedIn](http://links.otrenav.com/linkedin)  | [GitHub](http://links.otrenav.com/github) | [GitLab](http://links.otrenav.com/gitlab) | [CodeMentor](http://links.otrenav.com/codementor) |

---

# A Strategy for Predicting Ad Clicks with Unbalanced Data

- Written by Omar Trejo
- Written for client (JL, CM)
- January, 2017

The client approached me to help him work in an imbalanced problem regarding ad
clicks. He wants to understand how to properly sample data so that his models
produce reliable results. This document is a basic explanation on how to deal
with such problems and code is attached to exemplify.

## Project context

- Self-practice project
- For data science portfolio
- Background:
  - Economics degree
  - Computer Science experience
  - Data Science intensive course
  - Comfortable with mathematics and papers

## Technical context

- Objective: predict ad clicks
- Data:
  - 2 month random sample
  - 300 million observations
  - Variables:
    - `Unnamed 0`: Useless column that was added when merging
    - `SearchID` (categorical): ID for a visitors's search event
    - `AdID` (categorical): ID of the ad
    - `Position` (ordinal): position of the ad in search result page
    - `ObjectType` (categorical): type of the ad shown to user
    - `HistCTR` (numerical): past click-through rate for the ad
    - `IsClick` (boolean): 1 if there was a click on this ad
    - `SearchDate` (string): date and time of the search event
    - `UserID` (categorical): anonymized identifier of visitor's cookie
    - `IsUserLoggedOn` (categorical): whether user was logged in (1) or not (0)
    - `IPID` (categorical): anonymized identifier of visitor's IP
    - `SearchQuery` (string): raw query enter by user
    - `LocationID_x` (categorical): anonymized location ID where search was made
    - `LocationID_y` (categorical): ads geo-targeting
    - `CategoryID_x` (categorical): category filter of the search
    - `CategoryID_y` (categorical): the ads category according to the websites
    - `SearchParams` (json): number of search filters applied by the user
    - `Params` (json): features of the product shown in the Ad
    - `Price` (numerical): price of the ad
    - `Title` (string): raw title of the Ad
  - Current sample:
    - 484,506 observations
    - 0.6% event rate
- Current approach:
  - Logistic Regression
- Problems:
  - Intrinsic unbalanced data
    - Event reate: 0.6%
    - Can't get more observations

## Objectives

1. Learn how to do proper sampling
2. Learn how to do proper cross validation

## Suggested Strategies

When dealing with heavily imbalanced data you need to first understand if you
can assume that your data is representing the real distributions underlying the
problem. If you think that, even though you imbalanced data, the underlying
distributions are well represented, then the problem is manageable. A first step
is to find if the imbalance is intrinsic or extrinsic.

If the imbalance is extrinsic it means that there's something outside the
problem causing the imbalance (errors in the instruments used to collect data,
bureaucratic decisions that intentially filter certain data, and many others).
In these cases your first effort should be focused on getting more raw data to
accurately represent the underlying distributions. If the problem is intrinsic
(this project's case since ther actually are few clicks on ads) it means that
the problem itself produces imbalanced data (or you assume so because you
couldn't get more data), and there are a various approaches that may be useful
in such cases (assuming you have well represented distributions):

1. Sampling
2. Error weights
3. Feature engineering
4. Model engineering

These approaches have different conceptual levels. Deciding on error weights is
technically and conceptually easier than deciding on proper sampling techniques,
which in turn is easier than feature or model engineering.

Note that there are no significant cross validation techniques especifically
designed for imbalanced datasets. The part that changes when doing cross
validation is how you get your samples, which is part of the first approach
mentioned earlier. Therefore I will not mention cross validation specific
techniques here, and the standard approach is used.

### 1. Sampling

There are various techniques available for sampling imbalanced data, but there
are basically two fundamental concepts, and one that combines them:

1. Undersample the majority class
2. Oversample the minority class
3. Combination of both

A unordered list of such techniques included only for reference and without
explanation is:

- Random undersample
- Random oversample
- Clustered Centroids
- Near Miss 1, 2, and 3
- Condensed Nearest Neighbor
- Edited Nearest Neighbor
- Tomek Link Removal
- Adaptive Synthetic Sampling
- Synthetic Minority Oversampling

Each of these techniques has advantages and disadvantages and must be tested
when dealing with problems whose solution will actually be implemented. It's not
hard to come up with a sampling technique, and sometimes that's exactly what you
must do if your problem so requires.

For time restrictions in this specific project, I'll only show how to use the
Synthetic Minority Oversampling Technique (SMOTE) together with Edited Nearest
Neighbor (ENN), which are oversampling and undersampling techniques,
respectively. This means that we'll be using a combination of the two
fundamental sampling concepts mentioned earlier.

Again, for time restrictions, I will not go into the explanation of these
techniques and their advantages and disadvantages. I will only show you how to
use them (see the included code).

### 2. Error Weights

When you have heavily imbalanced data, as the one you're using for this project
(with a 0.6% even rate), the most accurate prediction you can do is fairly
obvious: predict that every event will be negative (i.e. no click on the ad). If
you do this, for the specific sample you sent me, you'll have an accuracy of
99.39%, which is very high.

The important part that you need to decide with your Type-I and Type-II error
weights (if you don't know what those are look them up, they are basic
statistics concepts for which you can find a lot of information online) is *how*
important is making a Type-I error (predicting an ad was clicked when it wasn't)
compared to your Type-II error (predicting an ad was not clicked when it was).

> Note: the actual definitions of Type-I and Type-II errors for your specific
> problem depend on what you decide your "positive" event to be, in this case
> I'm assuming positive event means that an ad was clicked.

A common way of specifying this error weights is using class weights within
`scikit-learn`, which basically defines how much weight you want to put on the
prediction accuracy for the different classes.

### 3. Feature engineering

Since this is not the focus of the current objectives I won't go deep into
feature engineering. I'll just explain the basics to not leave this important
part of dealing with imbalanced data out of the explanation.

When you have various variables for your imbalanced data, you may find that
there are a couple of them that are much more correlated with the positive event
you're trying to predict. When that's the case you're in luck and those will be
the strong rpredictors your models will be using. However, there are times that
there are no significant correlations among your variables and the positive
events you're looking for. That's when feature engineering comes into play.

Feature engineering basically means combining your existing variables to create
new variables that will, hopefully, exhibit more significant correlation with
the positive events you're looking for. These combinations may be experimental
(which may be very computational/time intensive since it's a combinatorial
problem, and as such as exponential complexity), but most of the time they come
from previous knowledge from the problem. For example it may be the case (this
is a total guess as I'm not an expert on ads) that an interaction between
position and the features of the product shown in the ad can provide better
information than they would provide separately.

Doing feature engineering is quite ellaborate and time consuming, requires
preprocessed/clean data, and requires domain expertise. All three things are
lacking for this project so I won't explore this further.

### 4. Model engineering

When working with real projects you seldom use a single model to solve a
problem. What you normally find is that different models predict better certain
characteristics of the problem that you're interested in. Normally unexperienced
people think they need to decide between one of those approaches. For example,
you may find that your Support Vector Machines generalized well but were bad at
catching some important outliers, while some Random Forests where good at
catching these outliers but overfitted quite easily. Then you may be tempted to
think that you must decide which of those is more important, and pick one
accordingly. The fact is that you can keep both!

What you do is add another level of abstraction. Meaning that the outputs of
each of those models (SVMs and RFs in the example I gave above) are turned into
the inputs of a new model which in turn makes the final decision (or may just as
easily be another intermediate layer). For these cases Logistic Regression is a
good model. These approaches are called Ensemble Methods. With imbalanced data
they are very useful to find different ways of extracting the information of the
rare samples.

Another thing to consider when doing model engineering is the loss function you
should use. This requires deep understanding of the data and domain knowledge to
do with meaningful arguments and is required only very specific cases.

Again, for time restrictions I won't create an ensemble, and I will just use a
Logistic Regression directly on the problem (what you were using) since the
request was about sampling, not about these other topics (feature and model
engineering), which I decided to briefly include because they're fundamental
parts of dealing with imbalanced data.

### A few words about Logistic Regression for imbalanced data

Logistic Regression does not require balanced data. Imbalanced data will affect
the regression's intercept which may distort the predicted probabilities but
it's not as bad as affecting the slope. This means that imbalanced data is not
catastrophic for Logistic Regression. However it's not a top choice for such
problems either. The reason being that highly imbalanced data often exhibit
significant non-linearities (in featured engineered spaces) that are not easily
learnable by linear models (Logistic Regression is a linear model).

Tree models are much better at learning these non-linearites (specifically
bagging or boosting techniques are prefered for this). However, Logistic
Regression is relatively good (much better than trees models) with small amounts
of data, which means that you can undersample your negative class (ads without
clicks) quite a bit and still achieve good results (see an example in the code).

I won't provide code for bagging or boosting techniques, but as a first step you
can try testing the problem with AdaBoost (a simple boosting technique). It
generally performs well in non-linear problems.

## Implementation

### Problems with the code you sent me

I understand the code you sent me was very simple and just to get a
proof-of-concepto working, but I thought it would be useful for you to know
what's wrong with that code. Here are the main points to notice:

1. You're using only one sample so you can't know how results differ among
   executions. This means that you can't be certain that the results are
   reliable or just a product of randomness in the sample chose. You need to
   test with various samples.

2. Even if you were taking various samples and using them to cross-validate the
   results, if the samples were just random samples, they would exhibit the same
   imbalanced characteristics. This can be improved by the sampling techniques
   mentioned above.

3. You're not specifying class weights which is very important for imbalanced
   problems. This means that your classifier is basically telling you to always
   predict that an add will not be clicked, which maximizes the prediction
   accuracy in a "dumb" way.

4. You're assuming that logarithmic loss is the best way to measure this
   problem. I don't know which measure is best for this specific problem (domain
   expertise is useful for this).

To fix these issues I introduce:

1. Stratified K-Fold Sampling
2. Sampling Techniques
2. Balanced Class Weights
3. Precision/Recall metrics

### 1. Stratified K-Fold Sampling

The objective of Stratified K-Fold Sampling is to keep the proportions in the
data being sampled for each random sample used for cross validation. That means
that if you have a sample that has a 10/1 ratio and you do simple random
sampling, you may end up with samples that have 10/0 or 10/3 ratios. This means
that when you train your models, the sample proportions (distributions) will
have changed. With imbalanced data this can potentially be a significant
problem, and it's avoidable by using this Stratified K-Fold Sampling to keep a
10/1 ratio for every sample you take from the original data.

In the code I created I'm using 5 samples every time we train the algorithm in
the `NUMBER_OF_SPLITS` constant. The actual Stratified K-Fold Sampling object is
in the `STRATIFIED_K_FOLD` object. These are in the `setup.py` file.

### 2. Sampling techniques

Stratified K-Fold Sampling helps with keeping the same underlying distrubutions
in the data for each new sample, but what if this underlying distributinos are a
problem themselves? With heavily imbalanced data this is a problem. The answer
is to use some of the techniques mentioned above to modify these underlying
distributions in the data. After having modifyied this underlying distributions
with some sampling techniques, Stratified K-Fold Sampling will ensure they are
kept intact for each new sample being taken from the modified data.

In the code I created I use two different scenarios one with Synthetic Minority
Oversampling Technique plus Edited Nearest Neighbor Undersampling (see the
`smoteenn.py` file), and one with Random Undersampling (see the
`undersample.py`). In the first a significant modification of the data is
performed by adding artificial/synthetic "new" samples and removing negative
samples around the same area. In the second a lot of negative observations are
removed to have balanced data without adding any artificial/synthetic
observations.

### 3. Balanced Class Weights

To really be able to measure the effects of an "intelligent" classifier as
opposed to a "dumb" classifier that just predicts a negative class for every
observation it sees due to the large imbalance in the data, we need to provide
`class_weights` to the algorithms. You can have this be part of a
cross-validation, but I just decided to use directly the
`class_weights='balanced'` parameter which makes sure that the weigthts are
inverse to the amount of data, meaning that there will be significant efforts to
classify positive events (ad clicks) properly instead of just trying to maximize
prediction accuracy.

### 4. Precision/Recall Metrics

To understand the difference in the approaches when classifying click on ads we
need to measure. We use the standard metrics: precision and recall. Precision is
the amount of predicted true positives divided by the sum of predicted true
positives and false positives. It's also known as specificity. Recall is the
amount of predicted true positives divided by the sum of true positive and false
negatives. It's also known as sensitivity. We would like to have both values as
high as possible.

Taking into account precision (specificity) and recall (sensitivity) is much
better to judge whether a model is good enough for an imbalanced problem than
just looking at the loss metric because it let's you make trade-off decisions.

## Results

For all the tests I used Stratified K-Fold Sampling with 5 splits. As we've seen
the uniform class weights are inadequate for this type of problems and are
included here only for reference of what you had before.

| Case | Class weights | Sampling Technique | Negative Precision | Positive Recall | Accuracy |
|------|---------------|--------------------|--------------------|-----------------|----------|
| 1 | Uniform  | Simple | 0.99 | 0.00 | 0.994159 |
| 2 | Balanced | Simple | 1.00 | 0.50 | 0.709073 |
| 3 | Balanced | SMOTE + ENN | 1.00 | 0.49 | 0.720058 |
| 4 | Balanced | RU | 1.00 | 0.54 | 0.659289 |

As can be seen there no one "perfect" solution. You have trade-offs that you
must decide when choosing a final model. In this case the best precision
accuracy comes from just saying that no ad will ever be clicked (case 1).
However, if predicting when an ad is actually clicked is important for you, you
may want to trade some predictive accuracy to increase your Positive Recall
metric, meaning that more clicked ads are predicted correctly. In real projects
the final decisions will come after lots of experimentation (much more than what
is shown here).

### Confusion matrices

#### Case 1

| Confusion matrix | Predicted Negative | Predicted Positive |
|------------------|--------------------|--------------------|
| Actually Negative | 144,503 | 0 |
| Actually Positive | 849 | 0 |

#### Case 2

| Confusion matrix | Predicted Negative | Predicted Positive |
|------------------|--------------------|--------------------|
| Actually Negative | 102,307 | 42,196 |
| Actually Positive | 421 | 428 |

#### Case 3

| Confusion matrix | Predicted Negative | Predicted Positive |
|------------------|--------------------|--------------------|
| Actually Negative | 104,246 | 40,257 |
| Actually Positive | 433 | 416 |

#### Case 4

| Confusion matrix | Predicted Negative | Predicted Positive |
|------------------|--------------------|--------------------|
| Actually Negative | 95,367 | 49,136 |
| Actually Positive | 387 | 462 |

### Sampling modifications

As can be seen, SMOTE + ENN bring the number of positive samples up while UR
makes the number of negative samples down. The important thing is *how* they do
it. To understand this look into the references. If you want, we can look at
these in another project or live session.

#### Case 3

| Samples | Negatives | Positives |
|---------|-----------|-----------|
| Before resample | 337,059 | 2,095 |
| After resample | 337,059 | 221,147 |

#### Case 4

| Samples | Negatives | Positives |
|---------|-----------|-----------|
| Before resample | 337,059 | 2,095 |
| After resample | 2,095 | 2,095 |

## Tools

- Python
- Scikit-learn
- Imbalance Learn

## Resources

- He & Ma, Imbalanced Learning, Foundations, Algorithms, and Applications,
  Wiley, 2013
- Chawla, Data minig for imbalanced datasets ()
- Chawla & et al, SMOTE, Synthetic Minority Over-Sampling Technique (2002)
- He & et al, ADASYN, Adaptive Synthetic Sampling Approach for Imbnalanced
  Learning (2008)

> Note: this references are not well formatted, but I think you can find them
> quite easily.

---

> "The best ideas are common property."
>
> —Seneca
