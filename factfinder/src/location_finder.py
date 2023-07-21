import random

import pandas as pd
import spacy
from fuzzywuzzy import fuzz
from spacy.tokens import DocBin
from spacy.training import Example
from tqdm import tqdm

tqdm.pandas()


def load_data(filename="clear_text_df.pkl"):
    # TODO: make available reading from different formats
    return pd.read_pickle(filename)


def get_similar_loc(row):
    str_a = row.splitted_text_tokenized
    str_b = str(row.street)
    words_lst = list()

    threshold = 80

    if isinstance(str_a, list):
        for word in str_a:
            if len(word) > 3:
                try:
                    for str_b_word in str_b.split(" "):
                        ratio = fuzz.partial_ratio(
                            str_b_word.lower(),
                            word.lower(),
                        )
                        if ratio > threshold:
                            words_lst.append(word)

                except AttributeError:
                    continue
    return words_lst


def get_loc(row):
    text = row.text
    words_lst = get_similar_loc(row)

    if not words_lst:
        return None

    for c, word in enumerate(words_lst):
        if c == 0 or c == len(words_lst) - 1:
            try:
                tmp_text = text.split(word, 1)
            except TypeError as e:
                print(e)
                return None
            except AttributeError as e:
                print(e)
                return None

            try:
                if c == 0:
                    start = len(tmp_text[0])
                if c == len(words_lst) - 1:
                    stop = len(text) - len(tmp_text[1])

            except IndexError as e:
                print(e)
                return None

    return start, stop, "street"


def train_spacy(data, iterations):
    for _, annotations in data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()

        for itn in range(iterations):
            print("Statring iteration " + str(itn))

            random.shuffle(data)
            losses = {}

            for text, annotations in tqdm(data):
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update(
                    [example],  # batch of texts
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses,
                )

            print(losses)

    return nlp


df = load_data()
df["entities"] = df.progress_apply(lambda row: get_loc(row), axis=1)

train_data = list()
train_df = df[df["entities"].notna()]

for i in tqdm(train_df.index):
    train_data.append(
        (train_df["text"][i], {"entities": [train_df["entities"][i]]})
    )

nlp = spacy.blank("ru")
doc_bin = DocBin()

if "ner" not in nlp.pipe_names:
    nlp.add_pipe("ner", last=True)

ner = nlp.get_pipe("ner")

for text, annot in tqdm(train_data):
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annot["entities"]:
        try:
            span = doc.char_span(
                start,
                end,
                label=label,
                alignment_mode="expand",
            )

            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)

        except IndexError:
            print("Skipping entity")
    try:
        doc.ents = ents  # label the text with the ents
        doc_bin.add(doc)
    except Exception as e:
        print(text, annot, e)

doc_bin.to_disk("train_2.spacy")

# TODO: make train_test_split
train = train_data[: int(len(train_data[:1000]) * 0.7)]
test = train_data[int(len(train_data[:1000]) * 0.7) :]

prdnlp = train_spacy(train, 100)
