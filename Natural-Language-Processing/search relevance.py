# code from kaggle competition on query result relevance

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




# -------------------------------- different approach


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

def str_stemmer(s):
    """
    function to stem the words of a sentense passed
    """
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
    """
    function to find common set of words between two sentenses passed and returning the count
    """
    return sum(int(str2.find(word)>=0) for word in str1.split())


# merging train and test, and mergeing another set of desc data for each product_id
df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, desc, how='left', on='product_uid')

# Feature extraction from our existing features 
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

# Dropping the features not required 
df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

df_train = df_all.iloc[:len(train)]
df_test = df_all.iloc[len(train):]
id_test = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values


from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
# Building our model on the product id and the engineered features

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

