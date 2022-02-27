#!/usr/bin/env python

import pytest

from framedct.framed_column_transformer import FramedColumnTransfomer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, impute
import pandas as pd
import numpy as np

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", impute.SimpleImputer(strategy="median")),
        ("scaler", preprocessing.StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", impute.SimpleImputer(strategy="most_frequent")),
        (
            "one_hot_encoder",
            preprocessing.OneHotEncoder(handle_unknown="error", drop="if_binary"),
        ),
    ]
)


@pytest.fixture
def data():
    """Sample pytest data."""
    return pd.DataFrame(
        {
            "Age": [5, 23, 16, 30, 45],
            "Height": [103, 185, 170, np.NaN, 175],
            "Gender": ["Male", "Female", "Female", "Male", np.NaN],
            "Country": ["Germany", "England", "Canada", "Canada", "France"],
        }
    )

num_features = ["Age", "Height"]
cat_features = ["Gender", "Country"]


def test_dataframe_output(data):
    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, num_features),
            ("categorical_pipeline", categorical_pipeline, cat_features),
        ]
    )

    transformed_data = ct.fit_transform(data)
    assert isinstance(transformed_data, pd.DataFrame)


def test_column_names(data):
    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, num_features),
            ("categorical_pipeline", categorical_pipeline, cat_features),
        ]
    )

    transformed_data = ct.fit_transform(data)
    assert transformed_data.columns.tolist() == [
        "Age",
        "Height",
        "Gender_Male",
        "Country_Canada",
        "Country_England",
        "Country_France",
        "Country_Germany",
    ]


def test_passthrough(data):
    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, num_features),
            ("categorical_pipeline", categorical_pipeline, ["Gender"]),
            ("remaining_features", "passthrough", ["Country"]),
        ]
    )

    transformed_data = ct.fit_transform(data)
    assert transformed_data.columns.tolist() == [
        "Age",
        "Height",
        "Gender_Male",
        "Country",
    ]

    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, num_features),
            ("remaining_features", "passthrough", ["Country"]),
            ("categorical_pipeline", categorical_pipeline, ["Gender"]),
        ]
    )

    transformed_data = ct.fit_transform(data)
    assert transformed_data.columns.tolist() == [
        "Age",
        "Height",
        "Country",
        "Gender_Male",
    ]


def test_drop(data):
    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, num_features),
            ("categorical_pipeline", categorical_pipeline, ["Gender"]),
            ("remaining_features", "drop", ["Country"]),
        ]
    )

    transformed_data = ct.fit_transform(data)
    assert transformed_data.columns.tolist() == ["Age", "Height", "Gender_Male"]


def test_remainder(data):
    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, num_features),
            ("categorical_pipeline", categorical_pipeline, ["Country"]),
        ],
        remainder="drop",
    )

    transformed_data = ct.fit_transform(data)
    assert transformed_data.columns.tolist() == [
        "Age",
        "Height",
        "Country_Canada",
        "Country_England",
        "Country_France",
        "Country_Germany",
    ]

    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, num_features),
            ("categorical_pipeline", categorical_pipeline, ["Country"]),
        ],
        remainder="passthrough",
    )

    transformed_data = ct.fit_transform(data)
    assert transformed_data.columns.tolist() == [
        "Age",
        "Height",
        "Country_Canada",
        "Country_England",
        "Country_France",
        "Country_Germany",
        "Gender",
    ]


def test_column_indexes(data):
    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, [0, 1]),
            ("categorical_pipeline", categorical_pipeline, [3]),
        ],
        remainder="drop",
    )

    transformed_data = ct.fit_transform(data)
    assert transformed_data.columns.tolist() == [
        "x0",
        "x1",
        "x0_Canada",
        "x0_England",
        "x0_France",
        "x0_Germany",
    ]

    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, [0, 1]),
            ("categorical_pipeline", categorical_pipeline, [3]),
        ],
        remainder="passthrough",
    )

    transformed_data = ct.fit_transform(data)
    assert transformed_data.columns.tolist() == [
        "x0",
        "x1",
        "x0_Canada",
        "x0_England",
        "x0_France",
        "x0_Germany",
        "Gender",
    ]

def test_delegation_to_column_transformer():
    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, num_features),
            ("categorical_pipeline", categorical_pipeline, cat_features),
        ]
    )

    assert ct.sparse_threshold == 0.3
    assert isinstance(ct.transformers, list)


def test_failed_delegation_to_column_transformer():
    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, num_features),
            ("categorical_pipeline", categorical_pipeline, cat_features),
        ]
    )

    with pytest.raises(AttributeError) as e_info:
        ct.no_such_method

def test_classification_pipeline(data):
    labels = [0, 1, 1, 0, 1]

    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, num_features),
            ("categorical_pipeline", categorical_pipeline, cat_features),
        ]
    )

    pipeline = Pipeline([
        ("transformer", ct),
        ("classifier", RandomForestClassifier())
    ])

    X_test = pd.DataFrame(
        {
            "Age": [15],
            "Height": [150],
            "Gender": ["Female"],
            "Country": ["Germany"],
        }
    )

    pipeline.fit(data, labels)
    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(X_test)

def test_regression_pipeline(data):
    labels = np.array([0, 10, 33, 45, 100])

    ct = FramedColumnTransfomer(
        transformers=[
            ("numerical_pipeline", numerical_pipeline, num_features),
            ("categorical_pipeline", categorical_pipeline, cat_features),
        ]
    )

    pipeline = Pipeline([
        ("transformer", ct),
        ("classifier", LinearRegression())
    ])

    X_test = pd.DataFrame(
        {
            "Age": [15],
            "Height": [150],
            "Gender": ["Female"],
            "Country": ["Germany"],
        }
    )

    pipeline.fit(data, labels)
    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(X_test)
