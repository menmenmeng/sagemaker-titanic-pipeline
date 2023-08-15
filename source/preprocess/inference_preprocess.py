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

from preprocess import *

if __name__ == "__main__":

    ###########################################################################################
    # Parsing
    ###########################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_output_dir', type=str, default="/opt/ml/processing/output")
    parser.add_argument('--base_preproc_input_dir', type=str, default="/opt/ml/processing/input")
    parser.add_argument('--split_rate', type=float, default=0.15)
    # parser.add_argument('--label_column', type=str, default="Survived") : inference용으로는 없어야 함

    # parse arguments
    args = parser.parse_args()

    base_preproc_input_dir = args.base_preproc_input_dir
    base_output_dir = args.base_output_dir
    split_rate = args.split_rate
    label_column = args.label_column

    # load titanic dataset
    df = pd.read_csv(base_preproc_input_dir + '/raw.csv')

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

    # ## label column 제거 : inference용 Preprocessing 코드이므로 없어야 함.
    # df_label = df[[label_column]]
    # df = drop_columns(df, [label_column])

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