import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    '''
    penalty, default="l2"
    tol, default=1e-4
    C, default=1.0

    '''
    parser.add_argument('--penalty', type=str, default="l2")
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--train_data_path', type=str, default="./data/output/train") # nb instance 내의 경로

    parser.add_argument('--model-dir', type=str, default="./data/output/model") # nb instance 내 경로
    parser.add_argument('--output-data-dir', type=str, default="./data/output") # nb instance 내 경로(이 파일에서 사용 X, cv 후 metric 저장을 위해)

    args = parser.parse_args()


    ###########################################################################################
    # dataset load
    ###########################################################################################

    data = pd.read_csv(f"{args.train_data_path}/train.csv") # nb instance 내 경로
    train = data.drop('Survived', axis=1)
    label = pd.DataFrame(data['Survived'])

    params = {
        'penalty': args.penalty,
        'tol': args.tol,
        'C': args.C
    }


    model = LogisticRegression().fit(train, label)
    joblib.dump(model, f'{args.model_dir}/model.pkl')
    # joblib.load(model.pkl)