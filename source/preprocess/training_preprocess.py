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

# from preprocess import *

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

if __name__ == "__main__":

    ###########################################################################################
    # Parsing
    ###########################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_output_dir', type=str, default="/opt/ml/processing/output")
    parser.add_argument('--base_preproc_input_dir', type=str, default="/opt/ml/processing/input")
    parser.add_argument('--split_rate', type=float, default=0.15)
    parser.add_argument('--label_column', type=str, default="Survived")

    # parse arguments
    args = parser.parse_args()

    base_preproc_input_dir = args.base_preproc_input_dir
    base_output_dir = args.base_output_dir
    split_rate = args.split_rate
    label_column = args.label_column

    # load titanic dataset
    df = pd.read_csv(base_preproc_input_dir + '/raw.csv', engine="python")

    # 컬럼 종류
    id_col = ['PassengerId']
    drop_cols = ['Name', 'Age', 'Ticket', 'Cabin', 'Embarked', 'Fare']
    one_hot_cols = ['Sex']



    ###########################################################################################
    # preprocessing pipeline
    ###########################################################################################

    ## id column 제거
    df_id = df[id_col]
    df = drop_columns(df, id_col)

    ## label column 제거
    df_label = df[[label_column]]
    df = drop_columns(df, [label_column])

    ## 결측치 column 제거
    df = drop_columns(df, drop_cols)

    ## one hot encoding
    df = one_hot_encode_COLUMNS(df, one_hot_cols)



    ###########################################################################################
    # 파이프라인 거쳐서 train, test 생성
    ###########################################################################################

    preprocessed_df = df
    preprocessed_df = convert_type(preprocessed_df, preprocessed_df.columns, type_target='float')
    preprocessed_df = pd.concat([df_id, df_label, preprocessed_df], axis=1)

    train_df, test_df = split_train_test(preprocessed_df, test_ratio=split_rate)
    train_df.to_csv(f"{base_output_dir}/train/train.csv", index=False)
    test_df.to_csv(f"{base_output_dir}/test/test.csv", index=False)