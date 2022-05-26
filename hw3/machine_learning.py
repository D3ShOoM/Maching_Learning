import numpy as np
import pandas as pd

def preprocess_classification_dataset():
    #train
    train_df = pd.read_csv('train.csv')
    train_feat_df = train_df.iloc[:,:-1] # grab all columns except the last one 
    train_output = train_df[['output']]
    X_train = train_feat_df.values 
    y_train = train_output.values
    #val
    val_df = pd.read_csv('val.csv')
    X_val = val_df.iloc[:,:-1].values
    y_val = val_df[['output']].values
    #test
    test_df = pd.read_csv('test.csv')
    X_test = test_df.iloc[:,:-1].values
    y_test = test_df[['output']].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def knn_classification(X_train, y_train, x_new, k=5):
    euc = [];
    m = len(X_train[0])
    for i in range(0, len(y_train)):
        under_root = 0
        for j in range(0,m):
            under_root += (X_train[i][j]-x_new[j])**2 
        euc.append(under_root**0.5)
    closest_ky = y_train[np.argpartition(euc, k)[:k]]
    values, counts = np.unique(closest_ky, return_counts=True)
    return values[np.argmax(counts)]

def logistic_regression_training(X_train, y_train, alpha=0.01, max_iters=5000, random_seed=1):
    ones = np.ones((len(X_train),1))
    X_copy = np.hstack((ones,X_train))

    np.random.seed(random_seed) # for reproducibility 
    weights = np.random.normal(loc=0.0, scale=1.0, size=(len(X_copy[0]), 1))
    
    for i in range(0, max_iters):
        weights = weights - (alpha*np.transpose(X_copy) @ (segmoid(X_copy, weights) - y_train))
    
    return weights


def segmoid(x,w):
    return 1/(1+np.exp(-1*(x @ w)))
    
    

def logistic_regression_prediction(X, weights, threshold=0.5):
    ones = np.ones((len(X),1))
    X_copy = np.hstack((ones,X))
    y_preds = segmoid(X_copy, weights)
    output = np.empty((len(y_preds),1))
    for i in range(len(y_preds)):
        output[i] = 0 if y_preds[i]<threshold else 1
    return output


def model_selection_and_evaluation(alpha=0.01, max_iters=5000, random_seed=1, threshold=0.5):
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_classification_dataset()
    
    preds_1nn = knn_predection(X_train, y_train, X_val, 1)
    preds_3nn = knn_predection(X_train, y_train, X_val, 3)
    preds_5nn = knn_predection(X_train, y_train, X_val, 5)
    
    weights = logistic_regression_training(X_train, y_train, alpha, max_iters, random_seed)
    preds_logistic = logistic_regression_prediction(X_val, weights, threshold)
    
    accuracy = np.empty(4)
    print('\n************************\n')
    print(len(preds_1nn))
    print('\n************************\n')
    print(len(y_val))
    print('\n************************\n')
    accuracy[0] = (y_val.flatten() == preds_1nn.flatten()).sum()/y_val.shape[0]
    accuracy[1] = (y_val.flatten() == preds_3nn.flatten()).sum()/y_val.shape[0]
    accuracy[2] = (y_val.flatten() == preds_5nn.flatten()).sum()/y_val.shape[0]
    accuracy[3] = (y_val.flatten() == preds_logistic.flatten()).sum()/y_val.shape[0]
    best = np.argmax(accuracy)
    
    methods = ('1nn', '3nn', '5nn','logistic regression')
    best_method = methods[best]
    
    X_train_val_merge = np.vstack([X_train, X_val]) 
    y_train_val_merge = np.vstack([y_train, y_val])
    
    if best == 0 :
        pred = knn_predection(X_train_val_merge, y_train_val_merge, X_test, 1)
    elif best == 1:
        pred = knn_predection(X_train_val_merge, y_train_val_merge, X_test, 3)
    elif best == 2:
        pred = knn_predection(X_train_val_merge, y_train_val_merge, X_test, 5)
    elif best ==3:
        w = logistic_regression_training(X_train_val_merge, y_train_val_merge, alpha, max_iters, random_seed)
        pred = logistic_regression_prediction(X_test, w, threshold)
    test_accuracy = (y_test.flatten() == pred.flatten()).sum() /y_test.shape[0]  
    
    return best_method, accuracy, test_accuracy


def knn_predection(X_train, y_train, X_val, k):
    preds = np.empty((len(X_val),1),dtype=int)
    for i in range(0, len(X_val)):
        preds[i] = knn_classification(X_train, y_train, X_val[i], k)
    return preds