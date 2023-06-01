import re

import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

tqdm.pandas()


class AddressExtractor:
    def __init__(self):
        self.score_threshold = 0.7
        self.address_len_threshold = 5
        self.street_names = [
            "улица",
            "проспект",
            "переулок",
            "шоссе",
            "бульвар",
        ]
        self.classifier = SequenceTagger.load(
            "Geor111y/flair-ner-addresses-extractor"
        )

    def _normalize_str(self, s):
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.findall(r"\d+|\D+", s)

        s_lst_tmp = list()
        for i in s:
            s_lst_tmp += i.split()

        s2_lst_tmp = list()
        for i in s_lst_tmp:
            if len(i) < 2 and not i.isdigit():
                pass
            else:
                s2_lst_tmp.append(i)

        s2_lst_tmp = " ".join(s2_lst_tmp)
        return s2_lst_tmp

    def _remove_street_names(self, s):
        for name in self.street_names:
            s = s.replace(name, "")
        return s

    def run(self, t):
        try:
            if t[0] == "[":
                t = t.split("], ", 1)[1]
        except Exception:
            return pd.Series([None, None, None])

        t = self._remove_street_names(t)
        t = self._normalize_str(t)

        if not t:
            return pd.Series([None, None, None])

        sentence = Sentence(t)
        self.classifier.predict(sentence)

        try:
            res = (
                sentence.get_labels("ner")[0]
                .labeled_identifier.split("]: ")[1]
                .split("/")[0]
                .replace('"', "")
            )
            score = round(sentence.get_labels("ner")[0].score, 3)

            if (
                score < self.score_threshold
                or len(res) < self.address_len_threshold
            ):
                return pd.Series([None, None, t])
            else:
                return pd.Series([res, score, t])

        except IndexError:
            return pd.Series([None, None, None])
