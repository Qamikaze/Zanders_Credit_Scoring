
from sklearn.utils import gen_batches, check_random_state

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer

import numpy as np

def random_feature_subsets(array, batch_size, random_state):
    """ Generate K subsets of the features in X """
    random_state = check_random_state(random_state)
    features = list(range(array.shape[1]))
    random_state.shuffle(features)
    for batch in gen_batches(len(features), batch_size):
        yield features[batch]


def augment(X, augmentation_matrix):
    return np.dot(X, augmentation_matrix)

def get_transformed_params(X, y, n_features_per_subset, max_epochs, random_state=777, augmentation_method='nn'):
    print(X.shape)
    print(y.shape)
    if random_state == 777:
        random_state = np.random.randint(0,700)
    transforming_params = []
    for i, subset in enumerate(random_feature_subsets(X, n_features_per_subset, random_state=random_state)):
        # if(augmentation_method=='nn')|(augmentation_method=='NN'):
        #     model, nn_history = nn_augment_model(X[:, subset], y, max_epochs = max_epochs)
        #     transforming_params.append((model, subset, nn_history))
        # else:
            if(augmentation_method == 'pca')|(augmentation_method == 'PCA'):
                pca = PCA()
                model = pca.fit(X[:, subset])
                transforming_params.append((model, subset))
            elif(augmentation_method=='rf')|(augmentation_method=='RF'):
                    rf = RandomForestRegressor(max_depth=5, random_state = 1,n_estimators=n_features_per_subset,
                    n_jobs=-1)
                    # rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                    #        class_weight='balanced_subsample', criterion='gini',
                    #        max_depth=5, max_features='sqrt', max_leaf_nodes=None,
                    #        max_samples=None, min_impurity_decrease=0.0,
                    #        min_impurity_split=None, min_samples_split=2, 
                    #        min_weight_fraction_leaf=0.0,
                    #        n_estimators=10, n_jobs=-1, oob_score=False,
                    #        random_state=1, verbose=0, warm_start=False)
                    #classnames, indices = np.unique(y, return_inverse=True)
                    model = rf.fit(X[:, subset], y)
                    transforming_params.append((model, subset))
            else:
                if (augmentation_method == 'rp') | (augmentation_method == 'RP'):
                    rp = GaussianRandomProjection(len(subset))
                    model = rp.fit(X[:, subset])
                    transforming_params.append((model, subset))
                else:
                    raise ValueError("`augmentation_method` must be `pca`, 'rp','rf' or `nn`, but was %s" % augmentation_method)
    return transforming_params


def get_transformed_matrix(X, transforming_params, augmentation_method):
    transformed_matrix = np.zeros(X.shape, dtype=np.float32)
    for tup in transforming_params:
        if(augmentation_method=='nn')|(augmentation_method=='NN'):
            model, subset, nn_history = tup
            transformed_matrix[np.ix_([True] * len(X), subset)] = model.predict(X[:, subset])
        elif(augmentation_method == 'rf')|(augmentation_method == 'RF'):
                model, subset = tup
                #transformed_matrix[np.ix_([True] * len(X), subset)] = FunctionTransformer(rf_apply, kw_args={"model": model}).transform(X[:, subset])
                transformed_matrix[np.ix_([True] * len(X), subset)] = model.apply(X[:,subset])
        else:
            if(augmentation_method == 'pca')|(augmentation_method == 'PCA')|(augmentation_method == 'rp')|(augmentation_method == 'RP'):
                model, subset = tup
                transformed_matrix[np.ix_([True] * len(X), subset)] = model.transform(X[:, subset])
            else:
                raise ValueError("`augmentation_method` must be `pca`, 'rp','rf or `nn`, but was %s" % augmentation_method)
    return transformed_matrix

def rf_apply(X, model):
    return model.apply(X)
