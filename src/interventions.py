import numpy as np

def intervene(x, i, t, X, Y, itype='class_mean'):
    """
        A single-feature counterfactual intervention of example x, using statistics
        from datatset (X,Y).

        Replaces i-th entry of x by a i-th entry of a statistic of (X,Y) (e.g. mean,
        median), which is computed either for the whole dataset (if scope is global)
        or for a certain class (pass as argument t).

        Arguments:
            x (ndarray): a (d,) array, the example to be intervened.
            i (int): the index of feature to be modified.
            t (int): Only relevant if scope is 'class'. This is the class whose statistic
                     will be used.
            X (ndarray): Dataset features, a (n,d) numpy array.
            Y (ndarray): Dataset labels, a (n,1) numpy array.
            itype (str): should be of the form 'scope_stat', where:
                - scope: One of ['class'|'global']. If 'class', will use only class t's
                         data to compute statistic. Otherwise, will use all data.
                - stat: A valid numpy statistic: ['mean'|'median'|'max'|'min']

    """
    x_hat = x.copy()
    if '_' in itype:
        scope, stat = itype.split('_')
    else:  # Default scope is global
        assert t is None, "target class provided but scope is global!"
        scope, stat = 'global', itype
    try:
        stat_fun = getattr(np, stat)  # retrieves callable
    except:
        raise ValueError('intervention stat not recognized')

    if scope == 'class':
        # TODO: We actually don't need all, just for t
        stats = {c: stat_fun(X[Y == c, :], axis=0) for c in np.unique(Y)}
        xi_hat = stats[t][i]
    elif scope == 'global':
        xi_hat = stat_fun(X, axis=0)[i]
    else:
        raise ValueError(
            'intervention scope not recognized (should be class_ or global_)')

    x_hat[i] = xi_hat
    return x_hat


def find_best_intervention(classifier, x, X, Y, itype='class_mean'):
    """
        Find best single-feature intervention which causes the biggest change
        in the classifier's prediction, measured as drop in P(X=c) for c the predicted
        class. Since we only onsider binary classifiers for now, this is equivalent
        to largest increase in P(X=c') for the non-predicted class c'.

        TODO: Generalize to multi-clas classifier.

        Arguments:
            classifier: a scikit-learn binary classifier or any object with methods
                        'predict_proba'and 'predict'.
            x (ndarray): a (d,) array, the example to be intervened.
            X (ndarray): Dataset features, a (n,d) numpy array.
            Y (ndarray): Dataset labels, a (n,1) numpy array.
            itype (str): should be of the form 'scope_stat', where:
                - scope: One of ['class'|'global']. If 'class', will use only class t's
                         data to compute statistic. Otherwise, will use all data.
                - stat: A valid numpy statistic: ['mean'|'median'|'max'|'min'].

    """
    init_p = classifier.predict_proba(x[None, :])[0]  # or squzee?
    init_c = classifier.predict(x[None, :])[0]
    trgt_c = 0 if init_c == 1 else 1
    best_delta = 0
    for feat in range(X.shape[1]):
        x_hat = intervene(x, feat, trgt_c, X, Y, itype)
        p_hat = classifier.predict_proba(x_hat[None, :])[0]
        # print(p_hat)
        if best_delta < (p_hat[trgt_c] - init_p[trgt_c]):
            best_delta = p_hat[trgt_c] - init_p[trgt_c]
            best_feat = feat
            best_x = x_hat
    return best_x, best_feat, best_delta
