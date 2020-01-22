import warnings
import pandas as pd
import numpy as np
import random

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from load_data import sc2_phenotype, sc2_outcome,sc2_cnv_data
from feature_selection import feature_selection_wrapper
from stack_ensemble import stack_ensemble_model


def sc2_model(cnv_data, phenotype, outcome):
    """
    Parameters
    ----------
    cnv_data:
        pandas DataFrame, DNA copy number and rows are samples, 
        columns are genomic cytobands.Values are reported as chromosome 
        instability values (CIN Index)
    phenotype:
        pandas DataFrame, One-Hot encoding to represent clinical phenotype
        [SEX, WHO_GRADING, CANCER_TYPE],rows are samples.
    outcome:
        pandas DataFrame, survival status outcome,A value of 0 means 
        the patient was alive or censoring at last follow up.A value 
        of 1 means patient died before the last scheduled follow up.
    Returns
    -------
    y_test:
        array, last time y_test.
    y_pred:
        array, last time y_pred.
    auc:
        float, last time cross validation auc
    model:
        Logistic Regression model train by all data
    feature:
        list, feature list
    """

    target = 'SURVIVAL_STATUS'
    data = pd.merge(outcome,
                    cnv_data,
                    left_index=True,
                    right_index=True)
    data = pd.merge(data,
                    phenotype,
                    left_index=True,
                    right_index=True)
    cnv_feature = list(cnv_data.var().sort_values(ascending=False)[:50].index)
    cnv_feature = feature_selection_wrapper(data[cnv_feature],
                                            data[target], 
                                            target, 
                                            10, 
                                            feature_return='log')
    feature = cnv_feature+list(phenotype.columns)
    
    x, y = data[feature].values.astype(float), data[target].values.astype(float)
    # Stratified ShuffleSplit cross-validator
    n = 20
    sp = StratifiedShuffleSplit(n_splits=n, test_size=0.5, random_state=0)
    tmp_auc = []
    # Stratified ShuffleSplit cross-validator
    for train_sample, test_sample in sp.split(x, y):
        print(len(tmp_auc))
        x_train, y_train = x[train_sample], y[train_sample]
        x_test, y_test = x[test_sample], y[test_sample]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf = stack_ensemble_model(x_train, y_train)

        print('test AUC',roc_auc_score(y_test,clf.predict_proba(x_test)[:,1]))
        y_pred = clf.predict(x_test)
        tmp_auc.append(roc_auc_score(y_test,clf.predict_proba(x_test)[:,1]))
        
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = stack_ensemble_model(x, y)
    print('average AUC',np.mean(tmp_auc))
    return y_test, y_pred, tmp_auc[-1], model, feature


if __name__ == '__main__':
    y_test, y_pred, auc, model, feature = sc2_model(sc2_cnv_data, sc2_phenotype, sc2_outcome)
    # Model Performance
    [[tn,fp],[fn,tp]] = confusion_matrix(y_test, 
                                         y_pred)
    #print confusion matrix
    print('T\\P\tAlive\tDied\tSum\nAlive\t{}\t{}\t{}\nDied\t{}\t{}\t{}\nSum\t{}\t{}\t{}\n'.format(tn,fp,tn+fp,
                                                                                        fn,tp,fn+tp,
                                                                                        tn+fn,fp+tp,tn+fp+fn+tp))
    print(classification_report(y_test, y_pred))
    print('Auc score',auc)
    print('Overall accuracy:',accuracy_score(y_test, y_pred,normalize=True))
    print('Feature list:{}'.format(feature))