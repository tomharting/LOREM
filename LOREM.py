# -*- encoding:utf-8 -*-
"""
This file is used to build and test a Language-consistent Open Relation Extraction Model (LOREM) from an existing
language-consistent and language-individual model.

Created by Tom Harting
"""

import LanguageIndividualModel as lim
from ProcessDataFile import create_index_label, create_index_ent
import numpy as np
import itertools
import warnings
from Evaluate import evaluation_rel
import json

# TOGGLE CPU/GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_probabilities(model_type, data_file, model_file, test_file, test_ent_file, other_test_language,
                      other_embedding_file):
    """
    Load the model, retrieve indices for the test files and compute the prediction probabilities for the model on
    the test data.

    :param model_type: The type of the loaded model (e.g. 'cnn-lstm')
    :param data_file: The path to the data file.
    :param model_file: The path to the model file.
    :param test_file: The path to the test file.
    :param test_ent_file: The path to the test entity file.
    :param other_test_language: Boolean which indicates if the test language is different than the training language.
    :param other_embedding_file: If the test language is different, this is the path to the embeddings file for this language.
    :return: Prediction probabilities of the model for the test set.
    """
    model, label_indices, max_sen_length, source_voc, target_voc, ent_voc = lim.load_model(model_type,
                                                                                           data_file,
                                                                                           model_file,
                                                                                           other_test_language,
                                                                                           test_file,
                                                                                           other_embedding_file)
    label_indices[0] = ''
    max_sen_length = 50

    test_data = create_index_label(test_file, max_sen_length, source_voc, target_voc)
    test_ent_data = create_index_ent(test_ent_file, max_sen_length, ent_voc)

    return lim.predict_labels(model, test_data, test_ent_data, label_indices), label_indices


def combine_probabilities(prob_ind, prob_con, lab_indices_ind, lab_indices_con, weight_ind):
    """
    Combine the probabilities of a language-individual and a language-consistent model.

    :param prob_ind: Probabilities of the language-individual model.
    :param prob_con: Probabilities of the language-consistent model.
    :param lab_indices_ind: Label indices of the language-individual model.
    :param lab_indices_con: Label indices of the language-consistent model.
    :param weight_ind: The weight factor of the language-individual probabilities (weight_con = 1 - weight_ind)
    :return: The combined probabilities.
    """
    if cmp(lab_indices_ind, lab_indices_con) != 0:
        warnings.warn('Warning: Models use different label indices!')

    prob_comb = []
    weight_con = 1. - weight_ind
    for batch_ind, batch_con in itertools.izip_longest(prob_ind, prob_con):
        for sentence_ind, sentence_con in itertools.izip_longest(batch_ind, batch_con):
            sentence_comb = []
            for word_ind, word_con in itertools.izip_longest(sentence_ind, sentence_con):
                word_comb = word_ind * word_con
                # Uncomment the line below to use weighted average combination approach instead of Hadamard product.
                # word_comb = weight_ind * word_ind + weight_con * word_con
                sentence_comb.append(word_comb.tolist())

            prob_comb.append(sentence_comb)

    return prob_comb


def prediction_to_labels(predictions, label_indices):
    """
    Convert prediction probabilities to relation labels.

    :param predictions: The prediction probabilities.
    :param label_indices: The indices of the relation labels.
    :return: The prediction result converted to relation labels.
    """
    pred_result = []
    for sentence in predictions:
        pred_sentence = []

        for word in sentence:
            index_pred_label = np.argmax(word)
            pred_label = label_indices[index_pred_label]
            pred_sentence.append(pred_label)

        result = []
        result.append(pred_sentence)
        pred_result.append(result)
    return pred_result


def combine_pred_and_truth(prediction, truth_file):
    """
    Combine the predicted labels and the ground truth labels for testing purposes.

    :param prediction: The prediction labels.
    :param truth_file: The ground truth file.
    :return: The combined prediction and ground truth labels.
    """
    f = open(truth_file, 'r')
    fr = f.readlines()
    prediction_and_truth = []

    for pred_labels, truth_line in itertools.izip(prediction, fr):
        instance = json.loads(truth_line.strip('\r\n'))
        truth_labels = instance['tags']
        prediction_and_truth.append([pred_labels[0], truth_labels])

    return prediction_and_truth


def clean_predictions(predictions):
    """
    Clean-up the predicted labels by changing impossible combinations (f.e. 'R-S', 'R-E' should be 'R-B', 'R-E').

    :param predictions: The predicted labels.
    :return: The cleaned predicted labels.
    """
    relation_labels = ['R-B', 'R-E', 'R-I', 'R-S']
    for line_index in range(0, len(predictions)):
        sentence = predictions[line_index][0]
        relation_started_flag = False
        for label_index in range(0, len(sentence) - 1):
            cur_label = sentence[label_index]
            upcoming_relations_flag = False

            if cur_label in relation_labels:
                for upcoming_label in sentence[label_index + 1:]:
                    if upcoming_label in relation_labels:
                        upcoming_relations_flag = True

                if relation_started_flag:
                    if upcoming_relations_flag:
                        cur_label = u'R-I'
                    else:
                        cur_label = u'R-E'
                else:
                    if upcoming_relations_flag:
                        cur_label = u'R-B'
                    else:
                        cur_label = u'R-S'
                    relation_started_flag = True

            predictions[line_index][0][label_index] = cur_label

    return predictions


def export_predictions(export_prediction_file, test_file, test_ent_file, predicted_result):
    """
    Export the predicted results to a text file.

    :param export_prediction_file: The file path to the export file.
    :param test_file: The sentences for which relations are predicted.
    :param test_ent_file: The corresponding entity pairs.
    :param predicted_result: The predicted relations.
    """
    export_file = open(export_prediction_file, 'w')
    test_file_read = open(test_file, 'r')
    test_lines = test_file_read.readlines()
    test_ent_file_read = open(test_ent_file, 'r')
    test_ent_lines = test_ent_file_read.readlines()

    for line_index in range(len(test_lines)):
        prediction = predicted_result[line_index][0]
        prediction = [tag for tag in prediction if tag != '']

        test_line = json.loads(test_lines[line_index].strip('\r\n'))
        sentence = test_line['tokens']

        test_ent_line = json.loads(test_ent_lines[line_index].strip('\r\n'))
        test_ent_tags = test_ent_line['tags']

        first_ent = ''
        second_ent = ''
        pred_rel = ''
        sent = ''

        for word, ent_tag, pred_tag in itertools.izip(sentence, test_ent_tags, prediction):
            sent += word + ' '

            if 'E1' in ent_tag:
                first_ent += word + ' '
            elif 'E2' in ent_tag:
                second_ent += word + ' '
            if 'R' in pred_tag:
                pred_rel += word + ' '

        export_file.write('Sent: ' + sent + '\n')
        export_file.write('Pred: < ' + first_ent + ', ' + pred_rel + ', ' + second_ent + '>\n')
        export_file.write('\n')

    export_file.close()
    test_file_read.close()
    test_ent_file_read.close()


if __name__ == "__main__":
    # Define the model types of the language-individual and the language-consistent models.
    model_type_individual = 'cnn_lstm'
    model_type_consistent = 'cnn_lstm'

    # Define the weight factor for the language-individual model (weight_consistent = 1 - weight_individual),
    # the weighted average is turned off in the combine_probabilities function by default.
    weight_individual = 0.5

    # Define the data and model files for the language-individual and the language-consistent models.
    data_file_individual = "./data/datafile/en_fasttext_wmorc_data.pkl"
    data_file_consistent = "./data/datafile/fr_sp_en_hi_ru_align_wmorc_data.pkl"

    model_file_individual = "./data/model/en_fasttext_wmorc_cnn_lstm.h5"
    model_file_consistent = "./data/model/fr_sp_en_hi_ru_align_wmorc_cnn_lstm.h5"

    # Define the test, test entity and test ground truth files.
    test_file = "../Data-generation/ClausIE-datasets/nyt_rel_test.json"
    test_ent_file = "../Data-generation/ClausIE-datasets/nyt_ent_test.json"
    test_truth_file = "../Data-generation/ClausIE-datasets/nyt_rel_test.json"

    # Define other word embedding file if the test file has a different language than the model.
    test_language_embedding = {'nl': "./data/embeddings/fastText/wiki.nl.align.vec"}

    # Define the file path of the prediction export file.
    export_prediction_file = "./export/wmorc_en_nyt_test.txt"

    # Option to output precision, recall and F1-score on the test set.
    test_option = False
    # Option to export prediction tuples from the test set to the export file.
    export_prediction_option = True
    # Option to clean the prediction (LOREMclean).
    clean_predictions_option = False
    # Options to indicate if the test language differs from the training language for both sub-models.
    other_test_language_individual = False
    other_test_language_consistent = False

    print('--++ Computing language-individual prediction ++--')

    probabilities_individual, label_indices_individual = get_probabilities(model_type_individual, data_file_individual,
                                                                           model_file_individual,
                                                                           test_file, test_ent_file,
                                                                           other_test_language_individual,
                                                                           test_language_embedding)

    print('--++ Computing language-consistent prediction ++--')
    probabilities_consistent, label_indices_consistent = get_probabilities(model_type_consistent, data_file_consistent,
                                                                           model_file_consistent,
                                                                           test_file, test_ent_file,
                                                                           other_test_language_consistent,
                                                                           test_language_embedding)

    print('--++ Combining language-individual and -consistent predictions ++--')
    probabilities_combined = combine_probabilities(probabilities_individual, probabilities_consistent,
                                                   label_indices_individual, label_indices_consistent,
                                                   weight_individual)

    print('--++ Converting predictions to labels ++--')
    predicted_result = prediction_to_labels(probabilities_combined, label_indices_individual)
    if clean_predictions_option:
        print('-- Cleaning up predicted labels --')
        predicted_result = clean_predictions(predicted_result)

    if test_option:
        print('--++ Evaluating prediction ++--')
        prediction_and_truth = combine_pred_and_truth(predicted_result, test_truth_file)

        P, R, F, PR_count, P_count, TR_count = evaluation_rel(prediction_and_truth)
        print('P= ', P, '  R= ', R, '  F= ', F)

    if export_prediction_option:
        print('--++ Exporting predictions ++--')
        print('Export file: ' + export_prediction_file)
        export_predictions(export_prediction_file, test_file, test_ent_file, predicted_result)
