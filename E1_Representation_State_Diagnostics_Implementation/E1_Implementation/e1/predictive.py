"""Grouped predictive analysis; requires scikit-learn for model evaluation."""
import numpy as np

def grouped_cv(X,y,groups,features=None,n_splits=5):
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    X=np.asarray(X,float);y=np.asarray(y,float);groups=np.asarray(groups)
    if X.ndim!=2 or len(X)!=len(y) or len(groups)!=len(y):raise ValueError('Mismatched rows')
    if not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Nonfinite inputs')
    unique=np.unique(groups)
    if len(unique)<n_splits:raise ValueError('Insufficient independent groups')
    results=[]
    for train,test in GroupKFold(n_splits=n_splits).split(X,y,groups):
        if set(groups[train])&set(groups[test]):raise AssertionError('Group leakage')
        model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
        model.fit(X[train],y[train]); pred=model.predict(X[test])
        null=np.full(len(test),y[train].mean())
        results.append({'n_train':len(train),'n_test':len(test),
                        'rmse':float(np.sqrt(mean_squared_error(y[test],pred))),
                        'null_rmse':float(np.sqrt(mean_squared_error(y[test],null)))})
    return results
