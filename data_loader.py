import pandas as pd
import numpy as np
from bert.tokenization.bert_tokenization import FullTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import emoji
import re
import random

def replace_digits_emojis(s):
    s = s.lower().strip()
    s = emoji.demojize(s)
    s = re.sub(r'\d+', '', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = s.strip()
    return s

def remove_urls_mentions(text):
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = text.replace("RT", "").strip()
    return text

def replace_space(text):
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", ' ', text)
    text = text.strip()
    return text

def merge_outputs(processed_text):
    text = ""
    for l in processed_text:
        if "</" in l:
            l = l.replace("</", "<")

        if l in ['<percent>', '<url>', '<', '<number>', '</allcaps>',
                     '<money>', '<phone>', '<allcaps>', '<repeated>',  '<hashtag>',
                      '<date>', '<time>', '<censored>', '</hashtag>', '<email>']:
            continue
        elif l in ['<emphasis>', '<user>', '<surprise>',  '<laugh>', '<sad>', '<annoyed>', '<happy>']:
            if l == '<user>':
                continue
            else:
                text += " " + l
        else:
            text += " " + replace_digits_emojis(l)
    normalized = replace_space(text)
    return normalized

def normalize_text(input_text:str, text_preprocessor):
    processed_text = text_preprocessor.pre_process_doc(input_text)
    normalized_text = merge_outputs(processed_text)

    return normalized_text

def sample_validation_set(X, y, ids):
    validation_sample_size = int((float(len(ids)) * 0.1)/2)

    X_train = {}
    y_train = {}
    y_train_ids = []
    X_valid = {}
    y_valid = {}
    y_valid_ids = []

    sampled_indexes = {0:[], 1:[]}
    index_counter = 0
    for label in y['output_label']:
        if len(sampled_indexes[label]) < validation_sample_size:
            sampled_indexes[label].append(index_counter)
        index_counter+=1


    for k in X:
        data = X[k]

        training_data = []
        validation_data = []
        index_counter = 0
        for d in data:
            label = y['output_label'][index_counter]

            # add to validation split
            if index_counter in sampled_indexes[label]:
                validation_data.append(d)
            else:
                training_data.append(d)

            index_counter +=1

        X_train[k] = np.array(training_data)
        X_valid[k] = np.array(validation_data)

    for k in y:
        data = y[k]

        training_data = []
        validation_data = []
        index_counter = 0
        for d in data:
            label = y['output_label'][index_counter]

            # add to validation split
            if index_counter in sampled_indexes[label]:
                validation_data.append(d)
            else:
                training_data.append(d)

            index_counter +=1

        y_train[k] = np.array(training_data)
        y_valid[k] = np.array(validation_data)

    index_counter = 0
    for id in ids:
        label = y['output_label'][index_counter]

        # add to validation split
        if index_counter in sampled_indexes[label]:
            y_valid_ids.append(id)
        else:
            y_train_ids.append(id)

        index_counter += 1

    return X_train, y_train, y_train_ids, X_valid, y_valid, y_valid_ids

def apply_oversampling(ids, labels, text_docs):

    count = {'HOF':0, 'NOT':0}
    label_to_ids = {'HOF':[], 'NOT':[]}

    c = 0
    for l in labels:
        count[l] +=1

        id = ids[c]
        label_to_ids[l].append(id)
        c+=1

    oversampled_ids, oversampled_labels, oversampled_text_docs = [], [], []

    if count['HOF'] > count['NOT']:
        max_label = 'HOF'
        min_label = 'NOT'
    else:
        max_label = 'NOT'
        min_label = 'HOF'


    label_diff = count[max_label] - count[min_label]

    random_ids = random.sample(label_to_ids[min_label], label_diff)

    for r in random_ids:
        id_index = ids.index(r)

        oversampled_ids.append(ids[id_index])
        oversampled_labels.append(labels[id_index])
        oversampled_text_docs.append(text_docs[id_index])

    # add the existing data
    oversampled_ids.extend(ids)
    oversampled_text_docs.extend(text_docs)
    oversampled_labels.extend(labels)

    return oversampled_ids, oversampled_labels, oversampled_text_docs

def tokenize(text):
    tags = ['<emphasis>', '<user>', '<surprise>', '<percent>', '<url>', '<', '<number>', '</allcaps>', '<money>',
                 '<phone>', '<allcaps>', '<repeated>', '<laugh>', '<hashtag>', '<elongated>', '<sad>', '<annoyed>',
                 '<date>', '<time>', '<censored>', '<happy>', '</hashtag>', '<email>']
    tokens = text.split(' ')
    filtered_tokens = []

    for t in tokens:
        if t not in tags:
            filtered_tokens.append(t)
    return filtered_tokens

def pad_text(max_seq_len, token_ids):
    token_ids = token_ids[:min(len(token_ids), max_seq_len - 2)]
    token_ids = token_ids + [0] * (max_seq_len - len(token_ids))
    return np.array(token_ids)

def embed_text_with_hate_words(config, data: list, hate_words: list):
    x = list()
    for text in data:

        # tokenize
        tokens = text.split(' ')
        multihot_encoding_array = np.zeros(len(hate_words), dtype=int)

        for t in tokens:
            if t in hate_words:
                index = hate_words.index(t)
                multihot_encoding_array[index] = 1

        x.append(multihot_encoding_array)
    return np.array(x)

def embed_text_with_bert(config: dict, data: list, bert_tokenizer: FullTokenizer):
    x = list()

    for text in data:

        # tokenize
        tokens = bert_tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        # convert tokens into IDs by embedding the text with BERT
        token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        # pad zeros to the token ids, if necessary
        max_seq_len = config['tweet_text_seq_len']
        token_ids = pad_text(max_seq_len, token_ids)
        x.append(token_ids)
    return np.array(x)

def embed_text_with_characters(config: dict, data: list):
    char_tokenizer = Tokenizer(lower=True, char_level=True, oov_token="UNKNOWN")

    alphabet = " abcdefghijklmnopqrstuvwxyz"
    char_dict = {"PADDING": 0, "UNKNOWN": 1}
    for i, char in enumerate(alphabet):
        char_dict[char] = len(char_dict)

    char_tokenizer.word_index = char_dict

    x = char_tokenizer.texts_to_sequences(data)

    x_padded = pad_sequences(x, padding='post', maxlen=config['tweet_text_char_len'])

    return x_padded

def normalize_text_docs(text_docs:list, text_preprocessor):
    normalized_text_docs = []
    for text in text_docs:
        normalized_text = normalize_text(text, text_preprocessor)
        normalized_text_docs.append(normalized_text)
    return normalized_text_docs

def encode_labels(data: list):
    y = list()
    label_to_index = {"HOF": 1, "NOT": 0}

    for label in data:
        y.append(label_to_index[label])
    return np.array(y)

def load_split(config, df, bert_tokenizer, hate_words, text_preprocessor, oversample:bool):
    X, y = {}, {}

    ids = df["id"].tolist()
    labels = df["label"].tolist()
    text_docs = df["text"].tolist()

    if oversample:
        ids, labels, text_docs = apply_oversampling(ids, labels, text_docs)

    if config['normalize_text']:
        text_docs = normalize_text_docs(text_docs, text_preprocessor)

    if "bert" in config["text_models"]:
        X["text_bert"] = embed_text_with_bert(config, text_docs, bert_tokenizer)
    if "hate_words" in config["text_models"]:
        X["text_hate_words"] = embed_text_with_hate_words(config, text_docs, hate_words)
    if "char_emb" in config["text_models"]:
        X["text_char_emb"] = embed_text_with_characters(config, text_docs)

    y['output_label'] = encode_labels(labels)
    return X, y, ids



def load_dataset(config, bert_tokenizer, hate_words):
    train_df = pd.read_csv(config['base_dir'] + 'resources/hasoc_data/'+config['dataset_year']+'/train.tsv', sep='\t', header=0)
    test_df = pd.read_csv(config['base_dir'] + 'resources/hasoc_data/'+config['dataset_year']+'/test.tsv', sep='\t', header=0)

    # load the Ekphrasis preprocessor
    text_preprocessor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
                  'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )

    X_train, y_train, y_train_ids = load_split(config, train_df, bert_tokenizer, hate_words, text_preprocessor, oversample=config['oversample'])
    X_test, y_test, y_test_ids = load_split(config, test_df, bert_tokenizer, hate_words, text_preprocessor, oversample=False)

    X_train, y_train, y_train_ids, X_valid, y_valid, y_valid_ids = sample_validation_set(X_train, y_train, y_train_ids)

    return X_train, y_train, y_train_ids, X_valid, y_valid, y_valid_ids, X_test, y_test, y_test_ids


