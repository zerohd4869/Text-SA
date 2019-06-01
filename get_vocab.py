import re
import sys

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords


def load_word_file(f_input):
    """
    Get all words in files
    #   {}dict无序  []list  ()tuple
    #   dict_1['key_1']  dict_2=fromkeys/update(dict_1)
    #   dict_1.get('key_1',default=None) #返回键的值 值不在返回None
    #   dict_1.setdefault('key_1',default=None)
    #   dict_1.items/keys/values()
    #
    """

    # with codecs.open(f_input, 'r', 'utf-8') as fr:
    f_input_pd = pd.read_csv(f_input, encoding="utf-8", engine='python')

    review_processed = f_input_pd['review'].map(processing_line)
    f_input_pd['review'] = review_processed

    file_words = {}
    with open('./kesci_data/corpus.vocab', 'w') as fp:
        for line in review_processed:
            fp.write(line)
            fp.write('\n')
            # 词频统计，存入file_words
            for word in line.split():
                file_words[word] = file_words.get(word, 0) + 1

    # if f_input_pd.isnull().sum().sum() != 0:
    # print(f_input+" total null count: %d" % f_input_pd.isnull().sum().sum())
    # f_input_pd = f_input_pd.dropna(axis=0)
    return f_input_pd, file_words


def get_vocab(path):
    """
        Get vocabulary file from the field 'postag' of files
    :param string: input train data file
    :param string: input dev data file
    """

    train_file_pd, word_dic = load_word_file(path[0][0])
    if len(word_dic) == 0:
        raise ValueError('The length of train word is 0')
    # train_data, validation_data = train_test_split(train_file_pd, test_size=0.2, random_state=0)
    # train_data.to_csv(path[1][0], index=None)
    # validation_data.to_csv(path[1][1], index=None)

    for i, test_file in enumerate(path[0][1]):
        test_data, dev_word_dic = load_word_file(test_file)
        if len(dev_word_dic) == 0:
            raise ValueError('The length of test%d word is 0' % i)
        for word in dev_word_dic:
            if word in word_dic:
                word_dic[word] += dev_word_dic[word]
            else:
                word_dic[word] = dev_word_dic[word]
        test_data.to_csv(test_file[:-4] + '_processed' + test_file[-4:], index=None)
    print('<unk>')  # 未登录词
    # vocab_set = set()
    # 按词频排序word_dic
    value_list = sorted(word_dic.items(), key=lambda d: d[1], reverse=True)
    freq_weight = 1
    for word in value_list[:int(len(value_list) * freq_weight)]:
        if word[1] > 1:
            print(word[0])
        # vocab_set.add(word[0])


def processing_line(line):
    line = line.lower().replace(',', ' ').replace('.', ' ').replace(':', ' ').replace('�', ' ').replace('?', '? ').replace('(', ' ( ').replace(')', ' ) ')
    if len(line.split()) > 2:
        # letters_only = re.sub("[^a-zA-Z]", " ", line)
        words = line.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
        line = " ".join(meaningful_words)
    return line


if __name__ == '__main__':
    """
    python get_vocab.py > ./kesci_data/train.vocab
    """

    train_file = './kesci_data/train_processed.csv'  # sys.argv[1]
    test_file_1, _2, _3 = './kesci_data/20190513_test.csv', './kesci_data/20190520_test.csv', './kesci_data/20190527_test.csv'
    new_train_file, new_valifation_file = './kesci_data/train_data.csv', './kesci_data/validation_data.csv'
    data_path = [[train_file, [test_file_1, _2, _3]], [new_train_file, new_valifation_file]]  # [raw_file_dir, new_file_dir]

    get_vocab(data_path)
