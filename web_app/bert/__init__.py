import os
from bert.bert_model import BERT

if not os.path.exists("bert/ml_models/bert_trained.pt"):
    raise OSError("Model not found")

bert_predictor = BERT()