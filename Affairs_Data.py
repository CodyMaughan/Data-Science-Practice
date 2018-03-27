import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

# Import data from statsmodels and add Affair classification
dta = sm.datasets.fair.load_pandas().data
dta['affair'] = (dta.affairs > 0).astype(int)

# Explore some correlation with affair and marriage rating by using pandas groupby and mean
print(dta.groupby(['affair']).mean())
print(dta.groupby(['rate_marriage']).mean())

fig = dta.educ.hist()
fig.set_title('Histogram of Education')
fig.set_xlabel('Education Level')
fig.set_ylabel('Frequency')
plt.show(block=True)

dta.rate_marriage.hist()
plt.title('Histogram of Marriage Rating')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')
plt.show(block=True)

pd.crosstab(dta.rate_marriage, dta.affair.astype(bool)).plot(kind='bar')
plt.title('Bar chart of multiple things')
plt.show()

affair_yrs_married_crosstab = pd.crosstab(dta.rate_marriage, dta.affair.astype(bool))
affair_yrs_married_crosstab.div(affair_yrs_married_crosstab.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Percentage based histogram')
plt.xlabel('Marriage Rating')
plt.ylabel('Affair Percentages')
plt.show()

affair_edu_crosstab = pd.crosstab(dta.educ, dta.affair.astype(bool))
affair_edu_crosstab.plot(kind='bar')
plt.title('Number of Affairs based on Education')
plt.xlabel('Education Level')
plt.ylabel('Frequency')
plt.show()

affair_edu_crosstab.div(affair_edu_crosstab.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Percentages of Affairs per Education Level')
plt.xlabel('Education Level')
plt.ylabel('Frequency')
plt.show()

# Now that we have visualized some things, lets prepare the data for logistic regression
y, X = patsy.dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
                  religious + educ + C(occupation) + C(occupation_husb)',
                  dta, return_type="dataframe")
print(X.columns.values)
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
                        'C(occupation)[T.3.0]':'occ_3',
                        'C(occupation)[T.4.0]':'occ_4',
                        'C(occupation)[T.5.0]':'occ_5',
                        'C(occupation)[T.6.0]':'occ_6',
                        'C(occupation_husb)[T.2.0]':'occ_husb_2',
                        'C(occupation_husb)[T.3.0]':'occ_husb_3',
                        'C(occupation_husb)[T.4.0]':'occ_husb_4',
                        'C(occupation_husb)[T.5.0]':'occ_husb_5',
                        'C(occupation_husb)[T.6.0]':'occ_husb_6'})

y = np.ravel(y)

# Now that the data is prepared, we can create the model
model = LogisticRegression()
model = model.fit(X, y)
print(model.score(X, y))
print(pd.DataFrame({'Features': np.ravel(X.columns), 'Coefs': np.ravel(np.transpose(model.coef_))}))

# Lets train with cross validation now
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)
predicted = model2.predict(X_test)

probs = model2.predict_proba(X_test)
print(probs)

# Here we have show model prediction metrics
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[:, 1]))

print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))

scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print(scores.mean())
print('----------')

# Now we are going to try and do some random forest classification
model3 = RandomForestClassifier(n_estimators=50, min_samples_split=10)
model3.fit(X_train, y_train)
print(model3.score(X_train, y_train))

predicted = model3.predict(X_test)
print(metrics.accuracy_score(y_test, predicted))
scores = cross_val_score(RandomForestClassifier(n_estimators=50, min_samples_split=10), X, y, scoring='accuracy', cv=10)
print(scores)
print(scores.mean())