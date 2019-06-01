import io
import os
import random

import numpy as np
import paddle
import paddle.fluid as fluid
import pandas as pd


# from inltk.inltk_main import setup
# setup('hi')
# from inltk.inltk_main import get_embedding_vectors


def get_predict_label(pos_prob):
    neg_prob = 1 - pos_prob
    # threshold should be (1, 0.5)
    neu_threshold = 0.55
    if neg_prob > neu_threshold:
        class3_label = 0
    elif pos_prob > neu_threshold:
        class3_label = 2
    else:
        class3_label = 1
    if pos_prob >= neg_prob:
        class2_label = 1  # 'Positive'
    else:
        class2_label = 0  # 'Negative'
    return class3_label, class2_label


def to_lodtensor(data, place):
    """
    convert ot LODtensor
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def data2tensor(data, place):
    """
    data2tensor
    """
    input_seq = to_lodtensor(list(map(lambda x: x[1], data)), place)
    return {"words": input_seq}


def data_reader(file_path, word_dict, is_shuffle=True):
    """
    Convert word sequence into slot
    """
    unk_id = len(word_dict)
    all_data = []
    with io.open(file_path, "r", encoding='utf8') as fin:
        for i, line in fin:
            cols = line.strip().split("\t")
            label = int(cols[0])
            wids = [word_dict[x] if x in word_dict else unk_id
                    for x in cols[1].split(" ")]
            all_data.append((wids, label))
    if is_shuffle:
        random.shuffle(all_data)

    def reader():
        for doc, label in all_data:
            yield doc, label

    return reader


def kesic_data_reader(file_path, word_dict, is_shuffle=True, max_data_len=200, max_words_len=200):
    """
    Convert word sequence into slot
    """
    unk_id = len(word_dict)
    all_data = []
    f_input_pd = pd.read_csv(file_path, encoding="utf-8", engine='python')
    f_input_df = pd.DataFrame(f_input_pd)
    id_list = f_input_df['ID']
    review_list = f_input_df['review']
    label_list = f_input_df['label'] if 'label' in f_input_df else []

    for i in range(len(id_list)):
        try:
            raw_id = int(id_list[i])
            label = int(1 if label_list[i] == 'Positive' else 0) if 'label' in f_input_df else int(2)
            if 'label' in f_input_df and label_list[i] not in ['Positive', 'Negative']:
                print("ERROR", i, id_list[i], review_list[i], label_list[i])
                continue

            raw_words = review_list[i].strip().split()
            wids = [word_dict[x] if x in word_dict else unk_id for x in raw_words]
            # words = [str(x) if x in word_dict else str('<unk>') for x in raw_words]

            if len(wids) > max_data_len:
                wids = wids[:max_data_len - 1]
            # if len(wids) > max_words_len:
            #     words = words[:max_words_len - 1]
        except KeyError and ValueError:
            print("ERROR", i, id_list[i], review_list[i], label_list[i])
            continue
        all_data.append((raw_id, wids, label))
    if is_shuffle:
        random.shuffle(all_data)

    test_size = int(len(all_data) * 0.2)

    def reader():
        for raw_id, doc_wids, doc_label in all_data:
            # text = get_embedding_vectors(' '.join(doc_text), 'hi')
            yield raw_id, doc_wids, doc_label

    return reader


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    # vocab_ = []
    with io.open(file_path, 'r', encoding='utf8') as f:
        wid = 0
        for i, line in enumerate(f):
            if i == 0: continue
            if line.strip() not in vocab:
                vocab[line.strip()] = wid
                # vocab_.append(line.strip())
                wid += 1
    vocab["<unk>"] = len(vocab)

    return vocab


def prepare_data(data_path, word_dict_path, batch_size, mode):
    """
    prepare data
    """
    assert os.path.exists(
        word_dict_path), "The given word dictionary dose not exist."
    if mode == "train":
        assert os.path.exists(
            data_path[0]) & os.path.exists(
            data_path[1]), "The given training data does not exist."
    if mode == "eval" or mode == "infer":
        assert os.path.exists(
            data_path), "The given test data does not exist."

    word_dict = load_vocab(word_dict_path)

    if mode == "train":
        train_reader = paddle.batch(kesic_data_reader(data_path[0], word_dict, True), batch_size)
        test_reader = paddle.batch(kesic_data_reader(data_path[1], word_dict, True), batch_size)
        return word_dict, train_reader, test_reader
    else:
        test_reader = paddle.batch(kesic_data_reader(data_path, word_dict, False), batch_size)
        return word_dict, test_reader
