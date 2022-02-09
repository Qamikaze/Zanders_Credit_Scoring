from xgboost import XGBModel, XGBClassifier #XGBoostLabelEncoder # laaste mss niet nodig

#from .callback import TrainingCallback
from typing import Sequence

from xgboost.compat import XGBModelBase
from xgboost import XGBClassifier
from sklearn.utils import check_random_state
from sklearn.utils import check_array

from sklearn.utils import check_X_y
from sklearn.utils import column_or_1d
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.tree.tree import DTYPE
from sklearn.metrics import log_loss
from typing import Union, Optional, List, Dict, Callable, Tuple, Any, TypeVar, Type, cast
from warnings import warn

from scipy.sparse import issparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd
import pickle
import datetime
import matplotlib.pyplot as plt
import xgboost as xgb
import scipy.special as ss

from XGBoostUtils import get_transformed_matrix, get_transformed_params

# deze nodig voor fit stages
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF

# nodig voor decision function
from sklearn.ensemble._gradient_boosting import predict_stages

_SklObjective = Optional[
    Union[
        str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ]
]

# import booster 
from xgboost.core import Booster, DMatrix, XGBoostError
from xgboost.training import train 


class XGBBase(XGBModel):

    # hier alle variabelen die mogelijk zijn toevoegen, nog bdenken of duplicate arguments andere naam geven, nu gwn verwijdert als dubbel
    def __init__(self, 
                 # deze van XGB Framework
                 n_estimators, max_depth, learning_rate, silent, objective, booster,
                 n_jobs, nthread, gamma, min_child_weight, max_delta_step,
                 subsample, colsample_bytree, colsample_bylevel, reg_alpha,
                 reg_lambda, scale_pos_weight, base_score, random_state,
                 seed, missing,
                 # dit van XGBoost + toegevoegde voor RF?
                 loss, criterion,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 min_impurity_decrease, min_impurity_split,
                 init, max_features,
                 n_features_per_subset, max_epochs, is_classification,
                 experiment_name='', augmentation_method='rf', trees_between_feature_update=10,
                 save_mid_experiment_accuracy_results=False, warm_start=False,
                 validation_fraction=0.1, tol=1e-4, max_leaf_nodes=None):
        
        # hier variablen die in XGBBase voorkomen
        super(XGBBase, self).__init__(n_estimators, max_depth, learning_rate, silent, objective, booster,
                                      n_jobs, nthread, gamma, min_child_weight, max_delta_step,
                                      subsample, colsample_bytree, colsample_bylevel, reg_alpha,
                                      reg_lambda, scale_pos_weight, base_score, random_state,
                                      seed, missing)

        self.booster = booster 
        self.n_estimators = n_estimators
        self.n_features_per_subset = n_features_per_subset
        self.is_classification = is_classification
        self.experiment_name = experiment_name
        self.augmentation_method = augmentation_method
        self.trees_between_feature_update = trees_between_feature_update
        self.save_mid_experiment_accuracy_results = save_mid_experiment_accuracy_results
        self.validation_fraction = validation_fraction
        self.warm_start = warm_start
        self.tol = tol
        self.alpha = 0.9
        self.max_epochs = max_epochs
        self.max_leaf_nodes = max_leaf_nodes
        self._SUPPORTED_LOSS = ('deviance', 'exponential', 'ls', 'lad', 'huber', 'quantile')
        self.ccp_alpha = 0.0
        self.loss = loss
        self.estimators_ = np.empty((0, 0), dtype=np.object)
        self.i = 0
        
        # dit toegevoegd voor initializen maar vaag of het nodig is
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        self.augmentations_ = []
        self.normalizer = QuantileTransformer(n_quantiles=1000, random_state=77)
        self.val_score_ = np.zeros((int(self.n_estimators),), dtype=np.float64)
        
        self.init = None
        self.init_ = None
          
    def _predict_stages(self, X):
        estimators = self.estimators_
        scale = self.learning_rate

        n_estimators = estimators.shape[0]
        K = estimators.shape[1]

        X_original = X
        X_normed = self.normalizer.transform(X)
        
        output = self.init_.predict(np.concatenate([X_original, X_original], axis=1)).astype(np.float64)        
        X_normed = self.normalizer.transform(X)

        if issparse(X):
            if X.format != 'csr':
                raise ValueError("When X is a sparse matrix, a CSR format is"
                                 " expected, got {!r}".format(type(X)))
        else:
            if not isinstance(X, np.ndarray) or np.isfortran(X):
                raise ValueError("X should be C-ordered np.ndarray,"
                                 " got {}".format(type(X)))
            for i in range(len(self.augmentations_)):
                X = get_transformed_matrix(X_normed, self.augmentations_[i], augmentation_method=self.augmentation_method)
                all_features_temp = np.concatenate([X_original, X], axis=1)
                tree_preds_to_add = np.zeros(output.shape)
                for k in range(K):
                    tree = estimators[i, k].tree_
                    predictions_temp = tree.predict(all_features_temp)
                    tree_preds_to_add[:, k] = scale * predictions_temp[:, 0]
                output += tree_preds_to_add
        return output


    def _init_state(self):
        self.augmentations_ = []
        self.normalizer = QuantileTransformer(n_quantiles=1000, random_state=77)
        self.val_score_ = np.zeros((int(self.n_estimators),), dtype=np.float64)
        self.i = 1
        self.estimators_ = np.empty((self.n_estimators, 1), dtype=np.object)
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        if self.init is None:
            self.init_ = super(XGBBase,self)
        else:
            self.init_ = self.init


    def _clear_state(self):
        if hasattr(self, 'val_score_'):
            del self.val_score_
        if hasattr(self, 'estimators_'):
            self.i = 0
            self.estimators_ = np.empty((0, 0), dtype=np.object)
        if hasattr(self, 'train_score_'):
            del self.train_score_   
        if hasattr(self, 'init_'):
            del self.init_    

    def _resize_state(self):
        self.val_score_.resize(self.n_estimators)
        # self.n_estimators is the number of additional est to fit
        total_n_estimators = self.n_estimators
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError('resize with smaller n_estimators %d < %d' %
                             (total_n_estimators, self.estimators_[0]))

        self.estimators_.resize((total_n_estimators, 1))
        self.train_score_.resize(total_n_estimators)
        
    def _is_initialized(self):
        return self.i > 0      


    """
    def decision_function(self, X):
        # deze array omzetting bestaat helemaal niet meer, dus probeer het zonder, hopleijk geen transformatie nodig
        #X = array2d(X, dtype=DTYPE, order="C")
        score = self._init_decision_function(X) # dit is dus gwn prediction van XGBboost
        predict_stages(self.estimators_, X, self.learning_rate, score) # geen idee wat hier doel van is/ werkt, maar lijkt ook niet echt iets te doen
        return score 
    """
    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, monitor=None):
        """Fit the AugBoost model.
            Parameters
            ----------
            X : array-like, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples
                and n_features is the number of features.
            y : array-like, shape = [n_samples]
                Target values (integers in classification, real numbers in
                regression)
                For classification, labels must correspond to classes.
            X_val : array-like, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples
                and n_features is the number of features.
            y_val : array-like, shape = [n_samples]
                Target values (strings or integers in classification, real numbers
                in regression)
                For classification, labels must correspond to classes.
            sample_weight : array-like, shape = [n_samples] or None
                Sample weights. If None, then samples are equally weighted. Splits
                that would create child nodes with net zero or negative weight are
                ignored while searching for a split in each node. In the case of
                classification, splits are also ignored if they would result in any
                single class carrying a negative weight in either child node.
            monitor : callable, optional
                The monitor is called after each iteration with the current
                iteration, a reference to the estimator and the local variables of
                ``_fit_stages`` as keyword arguments ``callable(i, self,
                locals())``. If the callable returns ``True`` the fitting procedure
                is stopped. The monitor can be used for various things such as
                computing held-out estimates, early stopping, model introspect, and
                snapshoting.
        Returns
        -------
        self : object
            Returns self.
        """

        # if not warmstart - clear the estimator state
        self._clear_state()
        
        # Check input
        
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], dtype=DTYPE)
        n_samples, b = np.concatenate([X, X], axis=1).shape

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)
 

        check_consistent_length(X, y, sample_weight)
        y = self._validate_y(y, sample_weight)

        sample_weight_val = None

        if not self._is_initialized():
            # init state
            self._init_state()
            mean = np.mean(y)
            y_pred = np.ones(len(y))
            y_pred = y_pred * mean
            begin_at_stage = 0

        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _decision_function (called in two lines
            # below) are more constrained than fit. It accepts only CSR
            # matrices.
            X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
            y_pred = self._decision_function(X) 
            self._resize_state()

        X_idx_sorted = None
        
        # deze functie lijkt beschikbaar, want zelf gedefinieerd
        n_stages = self._fit_stages(X, y, y_pred, sample_weight, 77,
                                    X_val, y_val, sample_weight_val,
                                    begin_at_stage, monitor, X_idx_sorted)

        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            self.val_score_ = self.val_score_[:n_stages]
            self.n_estimators_ = n_stages
        return self
    
    
    def gradient(self, predt: np.ndarray, y) -> np.ndarray:
        '''Compute the gradient squared log error.'''
        return (np.log1p(predt) - np.log1p(y)) / (predt + 1)
    
    def hessian(self, predt: np.ndarray, y) -> np.ndarray:
        '''Compute the hessian for squared log error.'''
        return ((-np.log1p(predt) + np.log1p(y) + 1) / np.power(predt + 1, 2))
     
    # HIer werd eerst wel ypred mee gegeven
    def _fit_stages(self, X, y, y_pred, sample_weight, random_state,
                    X_val, y_val, sample_weight_val,
                    begin_at_stage=0, monitor=None, X_idx_sorted=None):
        """Iteratively fits the stages.
        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """

        n_samples = X.shape[0]
        sample_mask = np.ones((n_samples,), dtype=np.bool)


        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        # perform boosting iterations
        i = begin_at_stage

        X_original = X
        from sklearn.preprocessing import Normalizer 
        X_normed = Normalizer().fit_transform(X)
        
        for i in range(begin_at_stage, 31):    

            print('training estimator #' + str(i))

            G = self.gradient(y_pred, y)
            H = self.hessian(y_pred, y)
            residuals = -0.5 * ( np.square(G) / (H + self.reg_lambda))
            
            print(residuals)
            if ((i % self.trees_between_feature_update) == 0):
                self.augmentations_.append(
                    get_transformed_params(X_normed, residuals, n_features_per_subset=self.n_features_per_subset,
                                           max_epochs=self.max_epochs, \
                                           random_state=i, augmentation_method=self.augmentation_method))
            else:
                self.augmentations_.append(self.augmentations_[-1])    
            X = get_transformed_matrix(X_normed, self.augmentations_[i], augmentation_method=self.augmentation_method)
            
            self.learning_rate = 0.3
            y_star = y_pred + self.learning_rate * residuals

            dmatrix = xgb.DMatrix(np.concatenate([X_original, X], axis=1), label=y_star)

            self.classifier.get_booster().update(dmatrix, i)
            y_pred = self.classifier.get_booster().predict(dmatrix)
            
            if (self.save_mid_experiment_accuracy_results):
                if self.is_classification:
                    y_val_preds = self.predict_proba(X_val)
                    y_train_preds = self.predict_proba(X_original)
                    self.val_score_[i] = log_loss(np.array(pd.get_dummies(y_val)), y_val_preds)
                    self.train_score_[i] = log_loss(pd.get_dummies(y), y_train_preds)                    
            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break

        if (self.save_mid_experiment_accuracy_results):
            self._save_results_and_figure_to_file()

        return i + 1

    def _save_results_and_figure_to_file(self):
        # saving sequence of losses to pickle, and saving this also plotted as a figure
        mid_process_results = {'train_score': self.train_score_[-1], 'val_score': self.val_score_[-1], \
                               'train_scores_sequence': self.train_score_, 'val_scores_sequence': self.val_score_, \
                               'experiment_time': str(datetime.datetime.now())[:19]}

        filename_for_writing = str(self.experiment_name) + '_' + str(self.max_epochs) + '_max_epochs_' + str(
            self.n_estimators) + '_trees_' + str(self.n_features_per_subset) + '_features_per_subset ' + \
                               mid_process_results['experiment_time']

        with open('results/' + filename_for_writing + '.pkl', 'wb') as f:
            pickle.dump(mid_process_results, f)

        plt.plot(mid_process_results['train_scores_sequence'])
        plt.plot(mid_process_results['val_scores_sequence'])
        plt.xlabel('# of tree in sequence')
        plt.ylabel('loss')
        plt.title('Train score: ' + str(mid_process_results['train_score'])[:5] + ', Val score: ' + str(
            mid_process_results['val_score'])[:5])
        plt.savefig('graphs/' + filename_for_writing + '.jpg')
        plt.close()

    def _decision_function(self, X):
        score = self._predict_stages(X)
        return score


    def _validate_y(self, y, sample_weight):
        self.n_classes_ = 1
        if y.dtype.kind == 'O':
            y = y.astype(np.float64)
        return y


    def _validate_X_predict(self, X, check_input):
        if check_input:
            X = self._validate_data(X, dtype=DTYPE, accept_sparse="csr", reset=False)
        return X   
   

    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        **check_params,
    ):
        X = check_array(X, **check_params)
        out = X
        return out
        

class XGBModelClassifier(XGBBase):
    """AugBoost for classification. AugBoost builds an additive model in a forward stage-wise fashion, while augmenting the features in between "boosts" using neural networks, PCA or random projections.
       Parameters
       ----------
       loss : {'deviance', 'exponential'}, optional (default='deviance')
           loss function to be optimized. 'deviance' refers to
           deviance (= logistic regression) for classification
           with probabilistic outputs. For loss 'exponential'
           this recovers the AdaBoost algorithm.
       learning_rate : float, optional (default=0.1)
           learning rate shrinks the contribution of each tree by `learning_rate`.
           There is a trade-off between learning_rate and n_estimators.
       n_estimators : int (default=100)
           The number of boosting stages to perform.
       max_depth : integer, optional (default=3)
           maximum depth of the individual regression estimators. The maximum
           depth limits the number of nodes in the tree. Tune this parameter
           for best performance; the best value depends on the interaction
           of the input variables.
       criterion : string, optional (default="friedman_mse")
           The function to measure the quality of a split. Supported criteria
           are "friedman_mse" for the mean squared error with improvement
           score by Friedman, "mse" for mean squared error, and "mae" for
           the mean absolute error. The default value of "friedman_mse" is
           generally the best as it can provide a better approximation in
           some cases.
           .. versionadded:: 0.18
       min_samples_split : int, float, optional (default=2)
           The minimum number of samples required to split an internal node:
           - If int, then consider `min_samples_split` as the minimum number.
           - If float, then `min_samples_split` is a fraction and
             `ceil(min_samples_split * n_samples)` are the minimum
             number of samples for each split.
           .. versionchanged:: 0.18
              Added float values for fractions.
       min_samples_leaf : int, float, optional (default=1)
           The minimum number of samples required to be at a leaf node:
           - If int, then consider `min_samples_leaf` as the minimum number.
           - If float, then `min_samples_leaf` is a fraction and
             `ceil(min_samples_leaf * n_samples)` are the minimum
             number of samples for each node.
           .. versionchanged:: 0.18
              Added float values for fractions.
       min_weight_fraction_leaf : float, optional (default=0.)
           The minimum weighted fraction of the sum total of weights (of all
           the input samples) required to be at a leaf node. Samples have
           equal weight when sample_weight is not provided.
       max_features : int, float, string or None, optional (default=None)
           The number of features to consider when looking for the best split:
           - If int, then consider `max_features` features at each split.
           - If float, then `max_features` is a fraction and
             `int(max_features * n_features)` features are considered at each
             split.
           - If "auto", then `max_features=sqrt(n_features)`.
           - If "sqrt", then `max_features=sqrt(n_features)`.
           - If "log2", then `max_features=log2(n_features)`.
           - If None, then `max_features=n_features`.
           Choosing `max_features < n_features` leads to a reduction of variance
           and an increase in bias.
           Note: the search for a split does not stop until at least one
           valid partition of the node samples is found, even if it requires to
           effectively inspect more than ``max_features`` features.
       max_leaf_nodes : int or None, optional (default=None)
           Grow trees with ``max_leaf_nodes`` in best-first fashion.
           Best nodes are defined as relative reduction in impurity.
           If None then unlimited number of leaf nodes.
       min_impurity_split : float,
           Threshold for early stopping in tree growth. A node will split
           if its impurity is above the threshold, otherwise it is a leaf.
           .. deprecated:: 0.19
              ``min_impurity_split`` has been deprecated in favor of
              ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
              Use ``min_impurity_decrease`` instead.
       min_impurity_decrease : float, optional (default=0.)
           A node will be split if this split induces a decrease of the impurity
           greater than or equal to this value.
           The weighted impurity decrease equation is the following::
               N_t / N * (impurity - N_t_R / N_t * right_impurity
                                   - N_t_L / N_t * left_impurity)
           where ``N`` is the total number of samples, ``N_t`` is the number of
           samples at the current node, ``N_t_L`` is the number of samples in the
           left child, and ``N_t_R`` is the number of samples in the right child.
           ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
           if ``sample_weight`` is passed.
           .. versionadded:: 0.19
       init : estimator, optional
           An estimator object that is used to compute the initial
           predictions. ``init`` has to provide ``fit`` and ``predict``.
           If None it uses ``loss.init_estimator``.
       warm_start : bool, default: False
           When set to ``True``, reuse the solution of the previous call to fit
           and add more estimators to the ensemble, otherwise, just erase the
           previous solution. See :term:`the Glossary <warm_start>`.
       max_epochs : int (default=10)
           Amount of maximum epochs the neural network meant for embedding will use each time
           (if early stopping isn't activated). Notice that a neural network will be trained a number of times,
           so be cautious with this parameter.
       save_mid_experiment_accuracy_results : bool (default=False)
           Whether to save the results from mid-experiment to file. The raw results will be saved as a pickle, and the graph will be saved as an image.
           If this is set as True, make sure you have a 'results' folder and a 'graphs' folder in the path from which you'll be running your experiment.
           This increases runtime significantly, therefore default is False.
       experiment_name : string (default='')
           Name of the experiment to be saved to file (details of the other parameters and timestamp will be included in the filename).
           The name should contain the dataset being tested, and also the unique charactersitics of the experiment.
       augmentation_method : string (default='nn')
           'nn', 'rp' or 'pca'. This augmentation will be used every 'trees_between_feature_update' amount of trees, to obtain new features from X.
           These features will be concatenated will the original features, and used for the next 'trees_between_feature_update' amount of trees.
           After that these featured will be "dumped" and new features will be created with the same method.
       trees_between_feature_update : int (default=10)
           This defines the frequency in which the features obtained by augmentation will be dumped, and new ones will be created.
           Notice that this parameter effects the run-time dramatically (both train and test).
       random_state : int, RandomState instance or None, optional (default=None)
           If int, random_state is the seed used by the random number generator;
           If RandomState instance, random_state is the random number generator;
           If None, the random number generator is the RandomState instance used
           by `np.random`.
       n_features_per_subset: int (default=4)
           number of features in each feature subset of the augmentation matrix.
           This is identical to the rotation for each tree in the rotation forest algorithm.
       validation_fraction : float, optional, default 0.1
           The proportion of training data to set aside as validation set for
           early stopping. Must be between 0 and 1.
           Only used if ``n_iter_no_change`` is set to an integer.
           .. versionadded:: 0.20
       tol : float, optional, default 1e-4
           Tolerance for the early stopping. When the loss is not improving
           by at least tol for ``n_iter_no_change`` iterations (if set to a
           number), the training stops.
           .. versionadded:: 0.20
       Attributes
       ----------
       n_estimators_ : int
           The number of estimators as selected by early stopping (if
           ``n_iter_no_change`` is specified). Otherwise it is set to
           ``n_estimators``.
           .. versionadded:: 0.20
       feature_importances_ : array, shape = [n_features]
           The feature importances (the higher, the more important the feature).
       oob_improvement_ : array, shape = [n_estimators]
           The improvement in loss (= deviance) on the out-of-bag samples
           relative to the previous iteration.
           ``oob_improvement_[0]`` is the improvement in
           loss of the first stage over the ``init`` estimator.
       train_score_ : array, shape = [n_estimators]
           The i-th score ``train_score_[i]`` is the deviance (= loss) of the
           model at iteration ``i`` on the in-bag sample.
           If ``subsample == 1`` this is the deviance on the training data.
       val_score_ : array, shape = [n_estimators]
           The i-th score ``val_score_[i]`` is the deviance (= loss) of the
           model at iteration ``i`` on the validation data.
           If ``subsample == 1`` this is the deviance on the validation data.
       nn_histories : list, shape not well defined
           Variable for saving the training history of all the NN's that were trained as part of the NetBoost model.
       loss_ : LossFunction
           The concrete ``LossFunction`` object.
       init_ : estimator
           The estimator that provides the initial predictions.
           Set via the ``init`` argument or ``loss.init_estimator``.
       estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, ``loss_.K``]
           The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
           classification, otherwise n_classes.
    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.
    ***This code is heavily based on scikit-learn's code repository (ensemble.gradient_boosting.py).
    This began as a fork of their code, and was modified to be an independent repository once we realized that it would be more convenient to use this way. Thanks!***
    """

    def __init__(# hier de inputs voor de base
                 self, max_depth=3, silent=True, booster='gbtree', nthread=None, gamma=0, subsample=1, seed=None, missing=None,
                 objective="binary:logistic",
                 min_child_weight=1, n_jobs=1,
                 max_delta_step=0, colsample_bytree=1,
                 colsample_bylevel=1, reg_alpha=0,
                 reg_lambda=1, scale_pos_weight=1, base_score=0.5,
                 random_state=0, learning_rate=0.1, n_estimators=100,
                 # hier de inputs voor Classifier, nv zelfde als base
                 # stond er al, hieronder zijn inputs voor RF en oude van augboost die wss overbodig zijn
                 loss='deviance',
                 criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 max_epochs=10, n_features_per_subset=4, max_features=None,
                 augmentation_method='nn', trees_between_feature_update=10,
                 max_leaf_nodes=None, warm_start=False,
                 validation_fraction=0.1,
                 experiment_name='AugBoost_experiment',
                 save_mid_experiment_accuracy_results=False, tol=1e-4):

        params = {'learning_rate': learning_rate, 'n_estimators': n_estimators, 
                 'booster': booster, 'silent': silent, 'nthread': nthread, 'seed':seed,
                 'gamma': gamma, 'min_child_weight': min_child_weight,
                 'max_delta_step': max_delta_step, 'subsample': subsample, 'colsample_bytree': colsample_bytree,
                 'colsample_bylevel': colsample_bylevel, 'reg_alpha': reg_alpha, 
                 'reg_lambda': reg_lambda, 'scale_pos_weight': scale_pos_weight, 'base_score': base_score,
                 'random_state': random_state, 'missing': missing, 'objective':objective,
                  'n_jobs':n_jobs, 'max_depth': max_depth,
                  # overig
                  'loss': loss, 
                  'criterion': criterion, 'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'min_weight_fraction_leaf': min_weight_fraction_leaf,
                  'init': init, 
                  'max_features': max_features,
                  'max_epochs': max_epochs,
                  'n_features_per_subset': n_features_per_subset,
                  'max_leaf_nodes': max_leaf_nodes,
                  'min_impurity_decrease': min_impurity_decrease,
                  'augmentation_method': augmentation_method,
                  'trees_between_feature_update': trees_between_feature_update,
                  'min_impurity_split': min_impurity_split,
                  'warm_start': warm_start, 
                  'validation_fraction': validation_fraction,
                  'experiment_name': experiment_name,
                  'save_mid_experiment_accuracy_results': save_mid_experiment_accuracy_results, 'tol': tol,
                  'is_classification': True}

        # nog niet helemaal zeker welke dit moeten zijn: kunnen veel meer maar geen zin
        params_xgb = {'learning_rate': learning_rate, 'n_estimators': n_estimators, 
                 'booster': booster,
                 'gamma': gamma, 'min_child_weight': min_child_weight,
                 'max_delta_step': max_delta_step, 'subsample': subsample, 'colsample_bytree': colsample_bytree,
                 'colsample_bylevel': colsample_bylevel, 'reg_alpha': reg_alpha, 
                 'reg_lambda': reg_lambda, 'scale_pos_weight': scale_pos_weight, 'base_score': base_score,
                 'random_state': random_state, 'missing': missing, 'objective':objective,
                  'n_jobs':n_jobs, 'max_depth': max_depth}
         
        super(XGBModelClassifier, self).__init__(**params)
        self.classifier = XGBClassifier(**params_xgb)


    def fit(self, X, y, sample_weight=None): 
        self.classifier.fit(np.concatenate([X, X], axis=1), y, sample_weight)
        super(XGBModelClassifier,self).fit(X, y, sample_weight=sample_weight) 

    def predict(self, X):   
        return self.classifier.predict(np.concatenate([X, X], axis=1))
    
    # predict proba 
    def predict_proba(self, X): 
        return self.classifier.predict_proba(np.concatenate([X, X], axis=1))
