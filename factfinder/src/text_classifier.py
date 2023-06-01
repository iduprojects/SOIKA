import pandas as pd
from transformers import pipeline


class TextClassifier:
    def __init__(
        self,
        repository_id="Sandrro/cc-model",
        number_of_categories=1,
        device_type=None,
    ):
        self.REP_ID = repository_id
        self.CATS_NUM = number_of_categories
        self.classifier = pipeline(
            "text-classification",
            model=self.REP_ID,
            tokenizer="cointegrated/rubert-tiny2",
            max_length=2048,
            truncation=True,
            device=device_type,
        )

    def run(self, t):
        preds = pd.DataFrame(self.classifier(t, top_k=self.CATS_NUM))
        self.classifier.call_count = 0
        if self.CATS_NUM > 1:
            cats = ", ".join(preds["label"].tolist())
            probs = ", ".join(preds["score"].round(3).astype(str).tolist())
        else:
            cats = preds["label"][0]
            probs = preds["score"].round(3).astype(str)[0]
        return [cats, probs]
