import os
import time
import json
import joblib
import pickle as pkl
import numpy as np
import pandas as pd
from io import BytesIO
NUM_FEATURES = 11


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    model = joblib.load(f"{model_dir}/model.pkl")
    return model


def input_fn(request_body, request_content_type):
    """
    The ScikitLearn LogReg model server receives the request data body and the content type,
    and invokes the `input_fn`.
    Return a nparray (an object that can be passed to predict_fn).
    """
    print("Content type: ", request_content_type)
    if request_content_type == "application/x-npy":        
        stream = BytesIO(request_body)
        array = np.frombuffer(stream.getvalue())
        array = array.reshape(int(len(array)/NUM_FEATURES), NUM_FEATURES)
        return array
    elif request_content_type == "text/csv":
        return pd.read_csv(request_body.rstrip("\n")).values()
    else:
        raise ValueError(
            "Content type {} is not supported.".format(request_content_type)
        )


def predict_fn(input_data, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.

    Return a two-dimensional NumPy array (predictions and scores)
    """
    start_time = time.time()
    y_probs = model.predict_proba(input_data)
    print("--- Inference time: %s secs ---" % (time.time() - start_time))    
    y_preds = model.predict(input_data)
    return np.vstack((y_preds, y_probs))


def output_fn(predictions, content_type="application/json"):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    if content_type == "text/csv":
        return ','.join(str(x) for x in outputs)
    elif content_type == "application/json":
        outputs = json.dumps({
            'pred': predictions[0,:].tolist(),
            'prob': predictions[1,:].tolist()
        })        
        
        return outputs
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))