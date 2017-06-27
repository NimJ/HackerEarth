# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:00:00 2017

@author: nimesh
"""

#import libraries
import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pickle


# start time of program         
start = time.time()
     

base_dir = 'C:\\Competitions\\Funding Successful Projects'


# Read Files
def read_files(base_dir):
    '''
    Read the input files: 
    '''
    train  = pd.read_csv('%s/train.csv' % base_dir)
    test  = pd.read_csv('%s/test.csv' % base_dir)
    return train,test
    

# Clean text
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower())+ ' '.join(emoticons).replace('-', '')
    return text

def impute_missing(df):
    for col in df.columns:
        if df[col].dtypes=='object':
            df[col].fillna('Unknown',inplace = True)
        else:
            df[col].fillna(-999,inplace = True)
    return df

# Dummy variable creation

def get_dummies(train, test):
    encoded = pd.get_dummies(pd.concat([train,test], axis=0))
    train_rows = train.shape[0]
    train = encoded.iloc[:train_rows, :]
    test = encoded.iloc[train_rows:, :] 
    return train,test


if __name__ == '__main__':
    
                    
    train,test = read_files(base_dir)
    
    # data processing & Missing value treatment
    
    y = train.final_status.values
    
    train = train.drop(['project_id','backers_count','final_status'], axis=1)
    test = test.drop(['project_id'], axis=1)
    
    #scaling
    scaler =  StandardScaler().fit(train['goal'])
    train['goal'] = scaler.transform(train['goal'])
    test['goal']  =  scaler.transform(test['goal'])
    
    train.disable_communication  = np.where(train.disable_communication==False,0,1)
    test.disable_communication  = np.where(test.disable_communication==False,0,1)
    
    # new feature
    train['duration_from_create'] = train['deadline'] - train['created_at']
    train['duration_from_launch'] = train['deadline'] - train['launched_at']
    train['state_change_from_create'] = train['state_changed_at'] - train['created_at']
    train['state_change_from_launch'] = train['state_changed_at'] - train['launched_at']
    train = train.drop(['deadline','created_at','state_changed_at','launched_at'],axis = 1)
    
    test['duration_from_create'] = test['deadline'] - test['created_at']
    test['duration_from_launch'] = test['deadline'] - test['launched_at']
    test['state_change_from_create'] = test['state_changed_at'] - test['created_at']
    test['state_change_from_launch'] = test['state_changed_at'] - test['launched_at']
    test = test.drop(['deadline','created_at','state_changed_at','launched_at'],axis = 1)
    
    #scaling
    scaler =  StandardScaler().fit(train['duration_from_create'])
    train['duration_from_create'] = scaler.transform(train['duration_from_create'])
    test['duration_from_create'] = scaler.transform(test['duration_from_create'])
    
    scaler =  StandardScaler().fit(train['duration_from_launch'])
    train['duration_from_launch'] = scaler.transform(train['duration_from_launch'])
    test['duration_from_launch'] = scaler.transform(test['duration_from_launch'])
    
    scaler =  StandardScaler().fit(train['state_change_from_create'])
    train['state_change_from_create'] = scaler.transform(train['state_change_from_create'])
    test['state_change_from_create'] = scaler.transform(test['state_change_from_create'])
    
    scaler =  StandardScaler().fit(train['state_change_from_launch'])
    train['state_change_from_launch'] = scaler.transform(train['state_change_from_launch'])
    test['state_change_from_launch'] = scaler.transform(test['state_change_from_launch'])
    
    #combine text columns
    train_text = train.apply(lambda x : '%s %s %s' %(x['name'],x['desc'],x['keywords']),axis = 1)
    test_text = test.apply(lambda x : '%s %s %s' %(x['name'],x['desc'],x['keywords']),axis = 1)
                                        
    train = train.drop(['name','desc','keywords'],axis = 1)
    test = test.drop(['name','desc','keywords'],axis = 1)
    
    #dummies for country and currency
    train,test = get_dummies(train,test)                                    
                                    
    #preprocess text data                       
    train_train = train_text.apply(lambda x : preprocessor(x))                                
    test_test = test_text.apply(lambda x : preprocessor(x))                                
#       
#    # train validation split 
#    X = train_text
#    kf = StratifiedKFold(y,round(1/.2),random_state = 123)
#    for train_indices,valid_indices in kf:
#        X_train,y_train = X.iloc[train_indices],y.iloc[train_indices]
#        X_valid,y_valid   = X.iloc[valid_indices],y.iloc[valid_indices]
#        
         
    tfidf = TfidfVectorizer(stop_words = 'english',token_pattern = r'\w{1,}',
                            strip_accents=None,
                            lowercase = False,preprocessor = None)
    
   
    
    
    train_text = tfidf.fit_transform(train_train)
    test_text = tfidf.transform(test_test)
   
    countv = CountVectorizer()
    train_count = countv.fit_transform(train_train)
    test_count = countv.transform(test_test)
     
    X_train = sparse.hstack((train_text,train_count,train),format='csr')
    X_test = sparse.hstack((test_text,test_count,test),format='csr')
#    
#        #presisting combined feature data
#    pickle.dump(X_train,open('%s/X_train_count.dat' %base_dir,'wb'))
#    pickle.dump(X_test,open('%s/X_test_count.dat' %base_dir,'wb'))
#    
#    #relaoding combined features
#    X_train = pickle.load(open('%s/X_train_count.dat' %base_dir,"rb"),encoding='latin1')
#    X_test = pickle.load(open('%s/X_test_count.dat' %base_dir,"rb"),encoding='latin1')
#    
    
    from sklearn.decomposition import TruncatedSVD
    svd =  TruncatedSVD(n_components = 120)
    X_train = svd.fit_transform(X_train)
    X_test = svd.transform(X_test)
    
    
    clf = GradientBoostingClassifier(n_estimators=500, 
                                     verbose = 1,
                                     subsample=0.7, 
                                     learning_rate= 0.01,
                                     max_depth=3,
                                     random_state=77)
    clf.fit(X_train, y)
    pred = clf.predict(X_test)
    
    sample  = pd.read_csv('%s/samplesubmission.csv' % base_dir)
    sample.final_status = pred    	
    sample.to_csv('%s/submission10.csv' % base_dir, index=False)

    
    print ('total time : %.2F Minutes' %((time.time()-start)/60) )
    
    # Model Persistance
    
#    import pickle
#    pickle.dump(clf,open('%s/classifier.pkl' %base_dir,'wb'),protocol=1)   
#    