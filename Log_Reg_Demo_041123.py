# import statements
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
import openpyxl

# logistic regression
marketing = pd.read_excel('C:/Users/ecriggins/Downloads/marketing_campaign.xlsx')
print(marketing.head().to_string())

# remove the rows with missing values/NA
marketing.dropna(inplace = True)

# print dimensions of the data frame
print(marketing.shape)

# data types
print(marketing.dtypes)

# change the data types where appropriate
marketing.Kidhome = marketing.Kidhome.astype(object)
marketing.Teenhome = marketing.Teenhome.astype(object)

# create our cat, numerical, and label/target arrays
X_num = marketing.iloc[:, 5:17].values # all numerical feature values
X_cat = marketing.iloc[:, 1:5].values # all categorical features, excluding the target
y = marketing.iloc[:, 0].values # label/target variable

print(X_num.shape)
print(X_cat.shape)
print(y.shape)

# use one-hot encoding/dummy coding for categorical variables
encoder = OneHotEncoder(sparse = False)
encoder.fit(X_cat)
X_enc = encoder.transform(X_cat)
print(f'Encoded Feature Array Shape {X_enc.shape}')

# combine numerical features with encoded features
X = np.hstack((X_num, X_enc))
print(X.shape)

# split into train/test 70/30
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = .30,
                                                    random_state = 1,
                                                    stratify = y)
print(X_train.shape)
print(X_test.shape)

# grid search for best parameters
lr_model = LogisticRegression()

# parameter grid
lr_params = {
    'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [1000, 1500, 2000],
    'multi_class': ['multinomial', 'auto', 'ovr'],
    'penalty': ['None', 'l1', 'l2']
}

# grid search with 10-fold CV
lr_grid = GridSearchCV(lr_model, lr_params, cv = 10,
                       refit = 'True', n_jobs = -1, verbose = 0,
                       scoring = 'accuracy')

# fit model to grid
lr_grid.fit(X_train, y_train) # always fit to the training set, not test

# store model with best params and accuracy
best_lr = lr_grid.best_estimator_

print('LR Best Parameters:', lr_grid.best_params_)
print('LR Best CV Score:', lr_grid.best_score_)
print('LR Training Acc:', best_lr.score(X_train, y_train))
print('LR Test Acc:', best_lr.score(X_test, y_test))


# confusion matrix
test_pred = best_lr.predict(X_test)
cm = confusion_matrix(y_test, test_pred)
cm_df = pd.DataFrame(cm, columns = ['0', '1'])
cm_df.index = ['0', '1']
print(cm_df)
print(classification_report(y_test, test_pred))
