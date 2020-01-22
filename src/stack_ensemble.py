from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def stack_ensemble_model(x, y):
    """Stack Ensemble
    Parameters
    ----------
    x:
        numpy array of shape [n_samples, n_features] Training set.
    y:
        numpy array of shape [n_samples] Target values.
    Returns
    -------
    model:
        model
    """

    #Logistic Regression parameter
    log_parameter_grid = {'class_weight' : ['balanced'],
                          'penalty' : ['l2'],
                          'solver': ['newton-cg', 'saga'],
                          'C' : [0.0001, 0.001, 0.01, 0.1]}
    #XGBoost Classifier parameter
    xgb_parameter_grid = {'max_depth':[2,3,4,5,6],
                          'class_weight' : ['balanced'],
                          'learning_rate':[0.1,0.2,0.3,0.4],
                          'n_estimators':[100,200,300,400,500]}
    #Random Forest Classifier parameter
    rf_parameter_grid = {'n_estimators': [100,200,300,400,500,600,700],
                         'class_weight' : ['balanced'],
                         'max_depth': [1,5,10]}
    #LinearSVC parameter
    svc_parmaeter_grid = {'penalty':['l2', 'l1'],
                          'class_weight' : ['balanced'],
                          'C': [0.1,0.2,0.3,0.4,0.5,1]}
    #===============GridSearch=======================================
    rf = GridSearchCV(RandomForestClassifier(random_state=42),
                      param_grid=rf_parameter_grid,
                      cv=20,
                      scoring='roc_auc',
                      n_jobs=-1,
                      refit=True)

    log = GridSearchCV(LogisticRegression(random_state=42),
                       param_grid=log_parameter_grid,
                       cv=20,
                       scoring='roc_auc',
                       n_jobs=-1,
                       refit=True)

    svc = GridSearchCV(LinearSVC(random_state=42),
                       param_grid=svc_parmaeter_grid,
                       cv=20,
                       scoring='roc_auc',
                       n_jobs=-1,
                       refit=True)

    xgb = GridSearchCV(XGBClassifier(random_state=42),
                       param_grid=xgb_parameter_grid,
                       cv=20,
                       scoring='roc_auc',
                       n_jobs=-1,
                       refit=True)
    #Fit grid search
    log.fit(x, y)
    rf.fit(x, y)
    svc.fit(x,y)
    xgb.fit(x,y)

        
    print('log Best score: {}'.format(log.best_score_))
    print('log Best parameters: {}'.format(log.best_params_))
    print('rf Best score: {}'.format(rf.best_score_))
    print('rf Best parameters: {}'.format(rf.best_params_))
    print('svc Best score: {}'.format(svc.best_score_))
    print('svc Best parameters: {}'.format(svc.best_params_))
    print('xgb Best score: {}'.format(xgb.best_score_))
    print('xgb Best parameters: {}'.format(xgb.best_params_))
    #==============================Stacking Ensemble=====================
    log = LogisticRegression(**log.best_params_)
    rf = RandomForestClassifier(**rf.best_params_)
    svc = LinearSVC(**svc.best_params_)
    xgb = XGBClassifier(**xgb.best_params_)

    estimators = [('rf', make_pipeline(rf)),
                  ('svc', make_pipeline(svc)),
                  ('xgb', make_pipeline(xgb)),
                  ('log', make_pipeline(log))]

    model = StackingClassifier(estimators=estimators, 
                               n_jobs=-1,
                               final_estimator=LogisticRegression(class_weight='balanced')).fit(x, y)
    return model