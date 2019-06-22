# -*- encoding:utf-8 -*-
"""
Functions which structure the input files as model input.

The code is written by Suncon Zheng and taken from https://github.com/TJUNLP/NSL4OIE. It is extended for multilingual
purposes by Tom Harting. Refactoring would be useful, but remains future work.
"""

import numpy as np
import pickle
import json


def load_vec_txt(files, vocab, k=300, embeddings_per_language=0, start_count=0, unknown_embedding=None):
    w2v = {}
    W = np.zeros(shape=(vocab.__len__() + start_count + 1, k))
    unknown_count = 0

    for lang, fname in files.iteritems():
        f = open(fname)
        emb_count = 0
        for line in f:
            if embeddings_per_language == 0 or emb_count < embeddings_per_language:
                values = line.split()
                word = values[0] + '_' + lang
                coefs = np.asarray(values[1:], dtype='float32')
                w2v[word] = coefs
                emb_count += 1
        f.close()

    if unknown_embedding is None:
        w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    else:
        w2v["UNK"] = unknown_embedding

    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
            unknown_count += 1
        W[vocab[word]] = w2v[word]

    print('Number of words with unknown embedding: ' + str(unknown_count))
    return w2v, k, W


def load_vec_onehot(vocab_w_inx):
    """
    Loads 300x1 word vecs from word2vec
    """
    k = vocab_w_inx.__len__()

    W = np.zeros(shape=(vocab_w_inx.__len__() + 1, k + 1))

    for word in vocab_w_inx:
        W[vocab_w_inx[word], vocab_w_inx[word]] = 1.
    # W[1, 1] = 1.
    return k, W


def create_index_word(file, max_s, source_vob, target_vob):
    """
    Coding the word sequence and tag sequence based on the digital index which provided by source_vob and target_vob
    :param the tag file: word sequence and tag sequence
    :param the word index map and tag index map: source_vob,target_vob
    :param the maxsent lenth: max_s
    :return: the word_index map, the index_word map, the tag_index map, the index_tag map,
    the max lenth of word sentence
    """

    data_s_all = []
    data_t_all = []
    f = open(file, 'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        t_sent = sent['tags']
        lang_sent = sent['language'][0]
        data_t = []
        data_s = []
        if len(s_sent) > max_s:

            # i=max_s-1
            # while i >= 0:
            #     data_s.append(source_vob[s_sent[i]])
            #     i-=1

            i = 0
            while i < max_s:
                word = s_sent[i] + '_' + lang_sent
                if not source_vob.__contains__(word):
                    data_s.append(source_vob["UNK"])
                else:
                    data_s.append(source_vob[word])
                i += 1
        else:

            # num=max_s-len(s_sent)
            # for inum in range(0,num):
            #     data_s.append(0)
            # i=len(s_sent)-1
            # while i >= 0:
            #     data_s.append(source_vob[s_sent[i]])
            #     i-=1

            i = 0
            while i < len(s_sent):
                word = s_sent[i] + '_' + lang_sent
                if not source_vob.__contains__(word):
                    data_s.append(source_vob["UNK"])
                else:
                    data_s.append(source_vob[word])
                i += 1
            num = max_s - len(s_sent)
            for inum in range(0, num):
                data_s.append(0)

        data_s_all.append(data_s)

        if len(t_sent) > max_s:
            for i in range(0, max_s):
                data_t.append(target_vob[t_sent[i]])
        else:
            for word in t_sent:
                data_t.append(target_vob[word])
            while len(data_t) < max_s:
                data_t.append(0)
        data_t_all.append(data_t)
    f.close()
    return [data_s_all, data_t_all]


def create_index_label(file, max_s, source_vob, target_vob):
    """
    Coding the word sequence and tag sequence based on the digital index which provided by source_vob and target_vob
    :param the tag file: word sequence and tag sequence
    :param the word index map and tag index map: source_vob,target_vob
    :param the maxsent lenth: max_s
    :return: the word_index map, the index_word map, the tag_index map, the index_tag map,
    the max lenth of word sentence
    """

    data_s_all = []
    f = open(file, 'r')
    fr = f.readlines()

    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        lang_sent = sent['language'][0]
        data_s = []
        if len(s_sent) > max_s:

            # i=max_s-1
            # while i >= 0:
            #     data_s.append(source_vob[s_sent[i]])
            #     i-=1

            i = 0
            while i < max_s:
                word = s_sent[i] + '_' + lang_sent
                if not source_vob.__contains__(word):
                    data_s.append(source_vob["UNK"])
                else:
                    data_s.append(source_vob[word])
                i += 1
        else:

            # num=max_s-len(s_sent)
            # for inum in range(0,num):
            #     data_s.append(0)
            # i=len(s_sent)-1
            # while i >= 0:
            #     data_s.append(source_vob[s_sent[i]])
            #     i-=1

            i = 0
            while i < len(s_sent):
                word = s_sent[i] + '_' + lang_sent
                if not source_vob.__contains__(word):
                    data_s.append(source_vob["UNK"])
                else:
                    data_s.append(source_vob[word])
                i += 1
            num = max_s - len(s_sent)
            for inum in range(0, num):
                data_s.append(0)

        data_s_all.append(data_s)

    f.close()
    return data_s_all


def create_index_ent(file, max_s, tag_vob):
    data_t_all = []
    f = open(file, 'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))

        t_sent = sent['tags']
        data_t = []

        if len(t_sent) > max_s:
            for i in range(0, max_s):
                data_t.append(tag_vob[t_sent[i]])
        else:
            for word in t_sent:
                data_t.append(tag_vob[word])
            while len(data_t) < max_s:
                data_t.append(0)
        data_t_all.append(data_t)
    f.close()

    return data_t_all


def get_word_index(train, test, start_count=1):
    """
    Give each word an index
    :param the train file and the test file
    :return: the word_index map, the index_word map, the tag_index map, the index_tag map,
    the max lenth of word sentence
    """
    source_vob = {}
    target_vob = {}
    target_labels = []
    sourc_idex_word = {}
    target_idex_word = {}
    count = start_count
    tarcount = 1
    # count = 0
    # tarcount = 0
    max_s = 0
    max_t = 0

    f = open(train, 'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tokens']
        lang = sent['language'][0]

        for word in sourc:
            word = word + '_' + lang

            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__() > max_s:
            max_s = sourc.__len__()

        target = sent['tags']

        if target.__len__() > max_t:
            max_t = target.__len__()
        for word in target:
            if word not in target_labels:
                target_labels.append(word)
                # target_vob[word] = tarcount
                # target_idex_word[tarcount] = word
                # tarcount += 1
    f.close()

    f = open(test, 'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tokens']
        lang = sent['language'][0]
        for word in sourc:
            word = word + '_' + lang
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__() > max_s:
            max_s = sourc.__len__()

        target = sent['tags']
        if not source_vob.__contains__(target[0]):
            source_vob[target[0]] = count
            sourc_idex_word[count] = target[0]
            count += 1
        if target.__len__() > max_t:
            max_t = target.__len__()
        for word in target:
            # if not target_vob.__contains__(word):
            #     target_vob[word] = tarcount
            #     target_idex_word[tarcount] = word
            #     tarcount += 1
            if word not in target_labels:
                target_labels.append(word)
    f.close()
    if not source_vob.__contains__("**END**"):
        source_vob["**END**"] = count
        sourc_idex_word[count] = "**END**"
        count += 1
    if not source_vob.__contains__("UNK"):
        source_vob["UNK"] = count
        sourc_idex_word[count] = "UNK"
        count += 1
    for label in sorted(target_labels):
        target_vob[label] = tarcount
        target_idex_word[tarcount] = label
        tarcount += 1

    return source_vob, sourc_idex_word, target_vob, target_idex_word, max_s


def get_entitylabeling_index(entlabelingfile):
    """
    Give each entity pair an index
    :param the entlabelingfile file
    :return: the word_index map, the index_word map,
    the max lenth of word sentence
    """
    ent_labels = []
    entlabel_vob = {}
    entlabel_idex_word = {}
    count = 1
    # count = 0
    max_s = 0

    f = open(entlabelingfile, 'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tags']
        for word in sourc:

            if word not in ent_labels:
                ent_labels.append(word)
                # entlabel_vob[word] = count
                # entlabel_idex_word[count] = word
                # print(count, ",,,,,,", entlabel_idex_word[count])
                # count += 1
        if sourc.__len__() > max_s:
            max_s = sourc.__len__()

    f.close()
    for label in sorted(ent_labels):
        entlabel_vob[label] = count
        entlabel_idex_word[count] = label
        count += 1

    return entlabel_vob, entlabel_idex_word, max_s


def create_data_file(train_file, test_file, word_embedding_files, model_file, train_ent_file, test_ent_file,
                     max_sen_length=50, embeddings_per_language = 0):
    """
    Converts the input files into the model input formats.

    :param train_file: The training file.
    :param test_file: The test file.
    :param word_embedding_files: The word embedding files.
    :param model_file: The model file.
    :param train_ent_file: The training entity file.
    :param test_ent_file: The testing entity file.
    :param max_sen_length: The maximum sentence length.
    :param embeddings_per_language: The maximum number of embeddings per language.
    """
    source_vob, sourc_idex_word, target_vob, target_idex_word, max_s = \
        get_word_index(train_file, test_file)

    print("Source vocabulary size: " + str(len(source_vob)))
    print("Target vocabulary size: " + str(len(target_vob)))
    print("Target vocabulary: " + str(target_idex_word))

    source_w2v, k, source_W = load_vec_txt(word_embedding_files, source_vob, embeddings_per_language=embeddings_per_language)

    print("Word embeddings loaded: " + str(word_embedding_files))
    print ("Word embeddings vocabulary size: " + str(len(source_w2v)))

    if max_s > max_sen_length:
        max_s = max_sen_length

    # Manual overwrite to equalize all languages for language-consistent model
    max_s = 50

    print ('Max sentence length: ' + str(max_s))

    train = create_index_word(train_file, max_s, source_vob, target_vob)
    test = create_index_word(test_file, max_s, source_vob, target_vob)

    entlabel_vob, entlabel_idex_word, entlabel_max_s = get_entitylabeling_index(train_ent_file)

    entlable_train = create_index_ent(train_ent_file, max_s, entlabel_vob)
    entlable_test = create_index_ent(test_ent_file, max_s, entlabel_vob)

    entlabel_k, entlabel_W = load_vec_onehot(entlabel_vob)


    out = open(model_file, 'wb')
    pickle.dump([train, test, source_W, source_vob, sourc_idex_word,
                 target_vob, target_idex_word, max_s, k,
                 entlable_train, entlable_test, entlabel_W, entlabel_vob, entlabel_idex_word], out)
    out.close()
    print ("Data file created.")
