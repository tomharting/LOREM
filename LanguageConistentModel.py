# -*- encoding:utf-8 -*-
"""
Create, train and test a language-consistent open relation extractor.

Created by Tom Harting
"""

from LanguageIndividualModel import train_model, test_model, load_model
from ProcessDataFile import create_data_file, create_index_ent, create_index_word
import random
import itertools

# TOGGLE CPU/GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def combine_data_files(all_files, all_ent_files, output_file, output_ent_file, lines_per_language=0):
    """
    Combine multiple single language data files.

    :param all_files: A list of all single language data files.
    :param all_ent_files: A list of all single language entity files.
    :param output_file: The path to the output file.
    :param output_ent_file: The path to the output entity file.
    :param lines_per_language: Number of lines that are randomly selected per language (by default all lines are used).
    """
    out_rel = open("./temp_rel_out.json", 'w')
    out_ent = open("./temp_ent_out.json", 'w')

    if lines_per_language == 0:
        for cur_file in all_files:
            data = open(cur_file, "r")
            out_rel.write(data.read())
            data.close()
        for cur_ent_file in all_ent_files:
            data = open(cur_ent_file, "r")
            out_ent.write(data.read())
            data.close()
    else:
        for cur_file, cur_ent_file in itertools.izip_longest(all_files, all_ent_files):
            cur_data = open(cur_file, "r")
            cur_ent_data = open(cur_ent_file, "r")
            cur_data_lines = cur_data.readlines()
            cur_ent_data_lines = cur_ent_data.readlines()

            data_size = len(cur_data_lines)
            remain_percentage = 100. * lines_per_language / data_size

            for sentence_index in range(0, data_size):
                rand = random.uniform(0, 100)
                if rand <= remain_percentage:
                    out_rel.write(cur_data_lines[sentence_index])
                    out_ent.write(cur_ent_data_lines[sentence_index])

    out_rel.close()
    out_ent.close()

    shuffle_languages(output_file, output_ent_file)


def shuffle_languages(output_file, output_ent_file):
    """
    Shuffle sentences from the temporary combined file and entity file in order to shuffle different
    languages after they were combined.

    :param output_file: The path to the output file.
    :param output_ent_file: The path to the output entity file.
    """
    input_rel = open("./temp_rel_out.json", 'r')
    input_ent = open("./temp_ent_out.json", 'r')
    input_data = input_rel.readlines()
    input_ent_data = input_ent.readlines()

    out_rel = open(output_file, 'w')
    out_ent = open(output_ent_file, 'w')

    data_size = len(input_data)
    shuffle_indices = range(0, data_size)
    random.shuffle(shuffle_indices)

    for sentence_index in shuffle_indices:
        out_rel.write(input_data[sentence_index])
        out_ent.write(input_ent_data[sentence_index])

    input_rel.close()
    input_ent.close()
    out_rel.close()
    out_ent.close()
    os.remove("./temp_rel_out.json")
    os.remove("./temp_ent_out.json")


if __name__ == "__main__":
    # Define the model type.
    model_type = 'cnn_lstm'

    # Define the data and model files.
    data_file = "./data/datafile/fr_sp_en_hi_ru_align_wmorc_data.pkl"
    model_file = "./data/model/fr_sp_en_hi_ru_align_wmorc_cnn_lstm.h5"

    # Define the train and test file paths.
    comb_train_file = "../Data-generation/WMORC/wmorc_fr_sp_en_hi_ru_rel_train.json"
    comb_train_ent_file = "../Data-generation/WMORC/wmorc_fr_sp_en_hi_ru_ent_train.json"
    comb_test_file = "../Data-generation/WMORC/wmorc_fr_sp_en_hi_ru_rel_test.json"
    comb_test_ent_file = "../Data-generation/WMORC/wmorc_fr_sp_en_hi_ru_ent_test.json"

    # Define the number of embeddings selected per language and the single language word embedding files.
    embeddings_per_language = 300000
    word_embedding_files = {'fr': "./data/embeddings/fastText/wiki.fr.align.vec",
                            'sp': "./data/embeddings/fastText/wiki.sp.align.vec",
                            'en': "./data/embeddings/fastText/wiki.en.align.vec",
                            'hi': "./data/embeddings/fastText/wiki.hi.align.vec",
                            'ru': "./data/embeddings/fastText/wiki.ru.align.vec"}

    # Define the number of lines selected per language and the single language train and test files.
    train_lines_per_language = 90000
    test_lines_per_language = int(train_lines_per_language * 0.2)
    all_train_file = ["../Data-generation/WMORC/wmorc_fr_rel_train.json",
                      "../Data-generation/WMORC/wmorc_sp_rel_train.json",
                      "../Data-generation/NeuralOIE/neural_oie_rel_train.json",
                      "../Data-generation/WMORC/wmorc_hi_rel_train.json",
                      "../Data-generation/WMORC/wmorc_ru_rel_train.json"]
    all_train_ent_file = ["../Data-generation/WMORC/wmorc_fr_ent_train.json",
                          "../Data-generation/WMORC/wmorc_sp_ent_train.json",
                          "../Data-generation/NeuralOIE/neural_oie_ent_train.json",
                          "../Data-generation/WMORC/wmorc_hi_ent_train.json",
                          "../Data-generation/WMORC/wmorc_ru_ent_train.json"]

    all_test_file = ["../Data-generation/WMORC/wmorc_fr_rel_test.json",
                     "../Data-generation/WMORC/wmorc_sp_rel_test.json",
                     "../Data-generation/NeuralOIE/neural_oie_rel_test.json",
                     "../Data-generation/WMORC/wmorc_hi_rel_test.json",
                     "../Data-generation/WMORC/wmorc_ru_rel_test.json"]
    all_test_ent_file = ["../Data-generation/WMORC/wmorc_fr_ent_test.json",
                         "../Data-generation/WMORC/wmorc_sp_ent_test.json",
                         "../Data-generation/NeuralOIE/neural_oie_ent_test.json",
                         "../Data-generation/WMORC/wmorc_hi_ent_test.json",
                         "../Data-generation/WMORC/wmorc_ru_ent_test.json"]

    # Option to output precision, recall and F1-score on the test set.
    test_option = True
    # Option to retrain the current model.
    retrain_option = False

    if not os.path.exists(data_file):
        print("--++ Creating data file ++--")

        if not os.path.exists(comb_train_file):
            print('Combining training and test files...')
            combine_data_files(all_train_file, all_train_ent_file, comb_train_file, comb_train_ent_file,
                               lines_per_language=train_lines_per_language)
            combine_data_files(all_test_file, all_test_ent_file, comb_test_file, comb_test_ent_file,
                               lines_per_language=test_lines_per_language)

        print('Creating data file...')
        create_data_file(comb_train_file, comb_test_file, word_embedding_files, data_file,
                         comb_train_ent_file, comb_test_ent_file, max_sen_length=50,
                         embeddings_per_language=embeddings_per_language)

    if not os.path.exists(model_file):
        print("--++ Training model ++--")
        print("Data file found: " + data_file)
        train_model(model_type, data_file, model_file, trainable_embeddings=True)
    else:
        if retrain_option:
            print("--++ Retraining model ++--")
            train_model(model_type, data_file, model_file, trainable_embeddings=True, retrain=retrain_option)

    if test_option:
        print("--++ Testing model ++--")

        model, target_index_word, result_file, max_sen_length, source_voc, target_voc, ent_voc = load_model(
            model_type, data_file, model_file)

        test_data = create_index_word(comb_test_file, max_sen_length, source_voc, target_voc)
        test_ent_data = create_index_ent(comb_test_ent_file, max_sen_length, ent_voc)

        P, R, F, PR_count, P_count, TR_count = test_model(model, test_data, test_ent_data, target_index_word)
        print('P= ', P, '  R= ', R, '  F= ', F)
