import numpy as np
import pandas as pd
from functools import wraps
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class FramedColumnTransfomer(BaseEstimator, TransformerMixin):
    """ A wrapper for scikit-learn `ColumnTransformer` that preserves Pandas `DataFrame` columns
        Parameters
            ----------
            transformers : list of tuples
                List of (name, transformer, columns) tuples specifying the
                transformer objects to be applied to subsets of the data.
                name : str
                    Like in Pipeline and FeatureUnion, this allows the transformer and
                    its parameters to be set using ``set_params`` and searched in grid
                    search.
                transformer : {'drop', 'passthrough'} or estimator
                    Estimator must support :term:`fit` and :term:`transform`.
                    Special-cased strings 'drop' and 'passthrough' are accepted as
                    well, to indicate to drop the columns or to pass them through
                    untransformed, respectively.
                columns :  str, array-like of str, int, array-like of int, \
                        array-like of bool, slice or callable
                    Indexes the data on its second axis. Integers are interpreted as
                    positional columns, while strings can reference DataFrame columns
                    by name.  A scalar string or int should be used where
                    ``transformer`` expects X to be a 1d array-like (vector),
                    otherwise a 2d array will be passed to the transformer.
                    A callable is passed the input data `X` and can return any of the
                    above. To select multiple columns by name or dtype, you can use
                    :obj:`make_column_selector`.
            remainder : {'drop', 'passthrough'} or estimator, default='drop'
                By default, only the specified columns in `transformers` are
                transformed and combined in the output, and the non-specified
                columns are dropped. (default of ``'drop'``).
                By specifying ``remainder='passthrough'``, all remaining columns that
                were not specified in `transformers` will be automatically passed
                through. This subset of columns is concatenated with the output of
                the transformers.
                By setting ``remainder`` to be an estimator, the remaining
                non-specified columns will use the ``remainder`` estimator. The
                estimator must support :term:`fit` and :term:`transform`.
                Note that using this feature requires that the DataFrame columns
                input at :term:`fit` and :term:`transform` have identical order.
            sparse_threshold : float, default=0.3
                If the output of the different transformers contains sparse matrices,
                these will be stacked as a sparse matrix if the overall density is
                lower than this value. Use ``sparse_threshold=0`` to always return
                dense.  When the transformed output consists of all dense data, the
                stacked result will be dense, and this keyword will be ignored.
            n_jobs : int, default=None
                Number of jobs to run in parallel.
                ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
                for more details.
            transformer_weights : dict, default=None
                Multiplicative weights for features per transformer. The output of the
                transformer is multiplied by these weights. Keys are transformer names,
                values the weights.
            verbose : bool, default=False
                If True, the time elapsed while fitting each transformer will be
                printed as it is completed.
            verbose_feature_names_out : bool, default=True
                If True, :meth:`get_feature_names_out` will prefix all feature names
                with the name of the transformer that generated that feature.
                If False, :meth:`get_feature_names_out` will not prefix any feature
                names and will error if feature names are not unique.
    """

    def __init__(
        self,
        transformers,
        *,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True
    ):
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out,
        )

    def __get_feature_out(self, estimator, features_in):
        if hasattr(estimator, "get_feature_names_out"):
            return estimator.get_feature_names_out(
                None if any(isinstance(e, int) for e in features_in) else features_in
            )
        elif estimator == "passthrough":
            return list(
                map(
                    lambda x: self.original_cols.get(x) if isinstance(x, int) else x,
                    features_in,
                )
            )
        elif estimator == "drop":
            return []
        else:
            return features_in

    def __get_pipeline_feature_names(self, pipeline, columns):
        current_cols = columns

        for step in pipeline:
            current_cols = self.__get_feature_out(step, current_cols)

        return current_cols

    def __get_transformer_feature_names(self, transformer, columns):
        return self.__get_feature_out(transformer, columns)

    def _get_feature_names(self):
        feature_names = []

        for _, estimator, columns in self.column_transformer.transformers_:
            if isinstance(estimator, Pipeline):
                feature_names.extend(
                    self.__get_pipeline_feature_names(estimator, columns)
                )
            else:
                feature_names.extend(
                    self.__get_transformer_feature_names(estimator, columns)
                )

        return feature_names

    def __getattr__(self, attr, *args, **kwargs):
        """As a wrapper to ColumnTransformer, we want to delegate everything that
        is not part of our interface"""

        if attr in dir(self):
            return super().__getattr__(attr, *args, **kwargs)
        elif attr in dir(self.column_transformer):
            return self.__bind_and_dispatch__(
                self.column_transformer, attr, *args, **kwargs
            )
        else:
            raise AttributeError(
                "'% s' object has no attribute '% s'" % (self.__class__.__name__, attr)
            )

    def __bind_and_dispatch__(self, receiver, attr, *args, **kwargs):
        """if `attr` is callable, wrap it and bind it, else, return the attribute's value"""
        method = getattr(receiver, attr)

        if callable(method):
            @wraps(method)
            def delegated_method(*args, **kwargs):
                return method(*args, **kwargs)

            setattr(self, attr, delegated_method)
            return delegated_method
        else:
            return method

    def fit(self, X, y=None):
        """Fit all transformers using X.
        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.
        y : array-like of shape (n_samples,...), default=None
            Targets for supervised learning.
        Returns
        -------
        self : FramedColumnTransfomer
            This estimator.
        """
        assert isinstance(X, pd.DataFrame)
        self.original_cols = {
            idx: col_name
            for col_name, idx in zip(X.columns, list(range(0, len(X.columns))))
        }

        self.column_transformer = self.column_transformer.fit(X, y=y)
        self.columns = self._get_feature_names()

        return self

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The data to be transformed by subset.
        Returns
        -------
        pd.DataFrame: DataFrame with the transformed data.
        """
        assert isinstance(X, pd.DataFrame)
        transformed_X = self.column_transformer.transform(X)

        if isinstance(transformed_X, np.ndarray):
            return pd.DataFrame(
                transformed_X,
                index=X.index,
                columns=self.columns
            )
        else:
            return pd.DataFrame.sparse.from_spmatrix(
                transformed_X,
                index=X.index,
                columns=self.columns
            )
