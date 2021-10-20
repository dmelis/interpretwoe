import numpy as np
import sklearn.datasets
import pandas as pd
from attrdict import AttrDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pdb

def load_boston_housing(binarize=True, test_size = 0.2):
    data = sklearn.datasets.load_boston()
    X = data['data']
    Y_cont = data['target']
    X_names = data['feature_names']

    if binarize:
        Y = pd.cut(Y_cont, [0,30,50], labels=['low', 'high'])
        Y_names = list(Y.categories)
        Y = Y.codes
    else:
        Y = Y_cont


    # Data Split
    ids = range(len(X))
    X_train, X_test, Y_train, Y_test, ids_train, ids_test = \
            train_test_split(X, Y, ids, test_size = test_size, random_state = 0)

    binary_feats = ['CHAS']

    cont_feats = np.array([not x in binary_feats for x in X_names])
    #Feature Scaling
    sc = StandardScaler()
    X_train[:,cont_feats] = sc.fit_transform(X_train[:, cont_feats])
    X_test[:,cont_feats]  = sc.fit_transform(X_test[:, cont_feats])

    X = AttrDict({'train': X_train, 'test': X_test, 'names': X_names})
    Y = AttrDict({'train': Y_train, 'test': Y_test, 'names': Y_names})

    # import pandas as pd
    df = pd.DataFrame(X.train, columns = X.names)

    return X,Y, df


# Online News Populatiry
# http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity

def load_online_news(target='channel', test_size = 0.2, seed= 2019, transform = None):

    fpath = '../data/raw/online_news/OnlineNewsPopularity.csv'

    df = pd.read_csv(fpath)
    df = df.rename(columns={a:a.strip() for a in df.columns})

    binary_feats = [cn for cn in df.columns if ('is_' in cn) or ('_was_' in cn)]
    lda_feats    = [cn for cn in df.columns if 'LDA' in cn]
    channel_feats = [x for x in df.columns if 'channel' in x]
    meta_feats   = ['url','timedelta']
    cont_feats = list(set(df.columns) - set(binary_feats) - set(meta_feats))
    drop_feats = meta_feats #binary_feats + # + lda_feats
    count_feats = [x for x in df.columns if 'kw' in x or 'num' in x or 'shares' in x or x.startswith('n_')]
    df = df.drop(columns=drop_feats)
    X_names = df.columns

    if target == 'shares':
        ### Task Version 1: Predict Shares
        Y = pd.cut(df['shares'], [0,500,1000,2000,5000,1000000], labels=['<500', '500-1000', '1000-2000','2000-5000','>5000'])
        Y_names = list(Y.cat.categories)
        Y = Y.cat.codes.to_numpy()
        df = df.drop(columns=['shares'])


    elif target == 'channel':
        ### Task Version 2: Predict Channel Category
        categories = np.array([a.split('_')[-1] for a in channel_feats])
        categories[categories == 'bus'] = 'business'

        # Some examples don't have any of these - fliter them out
        df = (df.loc[df[channel_feats].sum(axis=1) > 0]).reset_index()
        df['channel'] = pd.Categorical([categories[np.where(r==1)[0][0]] for r in df[channel_feats].to_numpy()], dtype='category')
        print('{:15} {}'.format('Class','Examples'))
        print(df['channel'].value_counts())

        Y = df.channel.cat.codes.to_numpy()
        Y_names = list(df.channel.cat.categories)

        df = df.drop(columns=channel_feats + ['channel'] + ['index'])

        count_feats += ['shares']

    X_names = df.columns#.to_list()

    df.loc[df['kw_min_min'] < 0,'kw_min_min'] = 0
    df.loc[df['kw_min_avg'] < 0,'kw_min_avg'] = 0
    df.loc[df['kw_avg_min'] < 0,'kw_avg_min'] = 0
    
    count_feats = list(set(count_feats))

    if transform  == 'log':
        df[count_feats] = df[count_feats].apply(lambda x: np.log(x+1.001))

    X_train, X_test, Y_train, Y_test = \
            train_test_split(df.to_numpy(), Y, test_size = test_size, random_state = seed)


    # #Feature Scaling
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    X = AttrDict({'train': X_train, 'test': X_test, 'names': X_names})
    Y = AttrDict({'train': Y_train, 'test': Y_test, 'names': Y_names})

    df_train = pd.DataFrame(X.train, columns = X.names).reset_index()
    # Create metafeats groups from features
    feature_idxs = [[i] for i in range(len(X.names))]
    metafeats_idxs = feature_idxs
    metafeats = {
        'length': [f for f in X.names if 'token' in f or 'n_non_stop' in f],
        'links':  [f for f in X.names if 'hrefs' in f or 'self' in f],
        #'keywords': [f for f in X.names if 'keywords' in f],
        'media': ['num_imgs', 'num_videos'],
        'keywords': [f for f in X.names if 'kw_' in f],
        #'self_refs':  [f for f in X.names if 'self' in f],
        'subjectivity': [f for f in X.names if 'subject' in f],
        'polarity': [f for f in X.names if ('polarity' in f or 'positive' in f or 'negative' in f)],
        'topic': [f for f in X.names if 'LDA' in f],
        'day': [f for f in X.names if 'week' in f],
        'channel': [f for f in X_names if 'channel' in f],
        'shares' : ['shares']
    }
    if target == 'channel': metafeats.pop('channel')
    if target == 'shares': metafeats.pop('shares')

    metafeats_idxs = [np.array([np.where(X.names == f)[0][0] for f in fs]) for a,fs in metafeats.items()]
    metafeats_names = list(metafeats.keys())
    metafeats_map = {feat: group for i,(group,feats) in enumerate(metafeats.items()) for feat in feats}

    metafeats_dict = AttrDict({'names':  metafeats_names , 'idxs': metafeats_idxs,
                               'groups': metafeats,
                               'mapping': metafeats_map})

    with open('../data/raw/online_news/feature_description.tsv') as f:
        rows = [l.strip().split('\t')for l in f]

    feat_desc = {row[0]:row[1] for row in rows}

    df_train = df_train.drop(columns='index')

    return X,Y, df_train, metafeats_dict, feat_desc
