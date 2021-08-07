import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg

train = pd.read_csv("/kaggle/working/train.csv")
test = pd.read_csv("/kaggle/working/test.csv")

# merging the text columns
traindata = list(train.apply(lambda x:'%s %s' % (x['product_title'],x['search_term']),axis=1))
testdata = list(test.apply(lambda x:'%s %s' % (x['product_title'],x['search_term']),axis=1))


from sklearn.metrics import mean_squared_error

#defining the loss function : RMSE - root mean squared error
def rms(actual, pred) :
    return mean_squared_error(actual, pred, squared=False)
  
# tfidf vectorizer
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

# Initialize SVD
svd = TruncatedSVD()
    
# Initialize the standard scaler 
scl = StandardScaler()
    
# We will use xgboost 

rf = xg.XGBRegressor(tree_method = 'gpu_hist') # for running on gpu 
    
# Create the pipeline 
clf = pipeline.Pipeline([
                ('tfv', tfv),   
                ('svd', svd),
                ('scl', scl),
#                 ('svm', svm_model)])
                ('rf', rf)])
    
# Create a parameter grid to search for best parameters for everything in the pipeline
param_grid = {'svd__n_components' : [200, 400]
#                   'svm__C': [10, 12]}
#               'rf__n_estimators': [500, 700]
             }
    
rms_scorer = metrics.make_scorer(rms)


model = GridSearchCV(estimator = clf, param_grid=param_grid, scoring=rms_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)


model.fit(traindata, y)
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    
    
    
# Initialize SVD
svd = TruncatedSVD(n_components= 200)
    
# Initialize the standard scaler 
scl = StandardScaler()
    
# We will use SVM here..

rf = xg.XGBRegressor(tree_method = 'gpu_hist')
    
# Create the pipeline 
clf = pipeline.Pipeline([
                ('tfv', tfv),   
                ('svd', svd),
                ('scl', scl),
#                 ('svm', svm_model)])
    ('rf', rf)])


# Fit model with best parameters optimized for quadratic_weighted_kappa
clf.fit(traindata,y)
preds = clf.predict(testdata)



