import json
import pathlib
import pickle

import joblib
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default="./data/output")    
    parser.add_argument('--model_path', type=str, default="./data/output/model/model.pkl")
    parser.add_argument('--test_path', type=str, default="./data/output/test/test.csv")
    parser.add_argument('--output_evaluation_dir', type=str, default="./data/output/evaluation")

    args = parser.parse_args()

    base_dir = args.base_dir
    model_path = args.model_path
    test_path = args.test_path
    output_evaluation_dir = args.output_evaluation_dir

    model = joblib.load(model_path)

    df = pd.read_csv(test_path)
    y_test = df[['Survived']]
    X_test = df.drop(['Survived'], axis=1)

    prediction = model.predict(X_test)
    pred_score = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, prediction)
    a1 = list([int(x) for x in cm[0]])
    a2 = list([int(x) for x in cm[1]])

    report_dict = {
        'pred_score' : pred_score,
        'cm' : {
            'a1' : a1,
            'a2' : a2
        }
    }

    evaluation_path = f"{output_evaluation_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))