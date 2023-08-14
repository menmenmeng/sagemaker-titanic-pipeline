import argparse
import os
import requests
import tempfile
import subprocess, sys

import pandas as pd
import numpy as np
from glob import glob

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_train_test(df, test_ratio=0.1):
    '''
    두 개의 데이터 세트로 분리
    '''
    total_rows = df.shape[0]
    train_end = int(total_rows * (1 - test_ratio))
    
    train_df = df[0:train_end]
    test_df = df[train_end:]
    
    return train_df, test_df


def convert_type(raw, cols, type_target):
    '''
    해당 데이터 타입으로 변경
    '''
    df = raw.copy()
    
    for col in cols:
        df[col] = df[col].astype(type_target)
    
    return df


def drop_columns(raw, drop_cols):
    '''
    열을 제거
    '''
    df = raw.drop(drop_cols, axis=1)
    return df


def one_hot_encode_ONECOLUMN(raw, one_hot_col):
    raw_ELSE = raw.drop([one_hot_col], axis=1)
    raw_OH = pd.get_dummies(raw[one_hot_col])
    df = pd.concat([raw_ELSE, raw_OH], axis=1)
    return df


def one_hot_encode_COLUMNS(raw, one_hot_cols):
    for col in one_hot_cols:
        raw = one_hot_encode_ONECOLUMN(raw, col)
    return raw