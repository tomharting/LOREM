# -*- encoding:utf-8 -*-
"""
Create, train and test a language-individual open relation extractor.

The original code is written by Suncon Zheng and taken from https://github.com/TJUNLP/NSL4OIE. It is substantially
refactored, commented and edited by Tom Harting.
"""


import os.path
import pickle
from shutil import copyfile

import h5py
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, \
    RepeatVector
from keras.layers.core import Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras_contrib import losses
from keras_contrib import metrics
from keras_contrib.layers import CRF

from Evaluate import evaluation_rel
from ProcessDataFile import create_data_file, create_index_ent, create_index_word, \
    get_word_index, load_vec_txt
# TOGGLE CPU/GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_training_input(input_sentences, input_ent_tags, input_labels, max_s, max_t, voc_size, target_index_word,
                       shuffle=False):
    """
    Process the training data to make it suitable for as input for the model.

    :param input_sentences: The training sentences.
    :param input_ent_tags: The entity tags of the training sentences.
    :param input_labels: The truth labels of the training sentences.
    :param max_s: The maximum sentence length.
    :param max_t: The maximum size of the target labels.
    :param voc_size: The size of the input vocabulary.
    :param target_index_word: The indexes of the target labels.
    :param shuffle: Option to shuffle the training set.

    :return: The formatted training input.
    """
    # Check if the number of sentences and ground truth labels are the same.
    assert len(input_sentences) == len(input_labels)

    # Sort and shuffle the sentences.
    indices = np.arange(len(input_sentences))
    if shuffle:
        np.random.shuffle(indices)

    input_sentences = input_sentences[indices]
    input_labels = input_labels[indices]
    input_ent_tags = input_ent_tags[indices]

    # Create empty arrays.
    train_sentences = np.zeros((len(input_sentences), max_s)).astype('int32')
    train_ent_tags = np.zeros((len(input_sentences), max_s)).astype('int32')
    train_labels = np.zeros((len(input_sentences), max_t, voc_size + 1)).astype('int32')

    # Fill the training arrays.
    for i, cur_sentence in enumerate(indices):
        train_sentences[i, ] = input_sentences[cur_sentence]
        train_ent_tags[i, ] = input_ent_tags[cur_sentence]
        for j, word in enumerate(input_labels[cur_sentence]):
            target_vec = np.zeros(voc_size + 1)

            if word != 0:
                word_str = target_index_word[word]

                if word_str.__contains__("O"):
                    target_vec[word] = 1
                else:
                    target_vec[word] = 1
            else:
                target_vec[word] = 1

            train_labels[i, j, ] = target_vec

    # Split training and validation set.
    num_validation_samples = int(0.2 * len(input_sentences))
    training_instances = train_sentences[:-num_validation_samples]
    training_labels = train_labels[:-num_validation_samples]
    training_entities = train_ent_tags[:-num_validation_samples]

    validation_instances = train_sentences[-num_validation_samples:]
    validation_labels = train_labels[-num_validation_samples:]
    validation_entities = train_ent_tags[-num_validation_samples:]

    return training_instances, training_entities, training_labels, validation_instances, validation_entities, \
           validation_labels


def save_model(model, model_file_path):
    """
    Save the model weights.

    :param model: The model.
    :param model_file_path: The file path where the weights are stored.
    """
    model.save_weights(model_file_path, overwrite=True)


def create_cnn_lstm_model(source_voc_size, target_voc_size, ent_voc_size, source_words,
                          ent_voc, input_seq_length,
                          output_seq_length,
                          hidden_dim, emd_dim, trainable_embeddings=False):
    """
    Construct a CNN-BiLSTM open relation extraction model.

    :param source_voc_size: The size of the source vocabulary.
    :param target_voc_size: The size of the target vocabulary.
    :param ent_voc_size: The size of the entity tag vocabulary.
    :param source_words: The input word embeddings.
    :param ent_voc: The entity tag vocabulary.
    :param input_seq_length: The maximum input sentence length.
    :param output_seq_length: The output sequence length.
    :param hidden_dim: The dimensionality of the hidden layers.
    :param emd_dim: The dimensionality of the word embeddings.
    :param trainable_embeddings: Option to optimize the word embeddings during training.

    :return: A non-trained open relation extraction model.
    """
    # Define the model input.
    word_input = Input(shape=(input_seq_length,), dtype='int32')
    ent_input = Input(shape=(input_seq_length,), dtype='int32')

    word_embedding_lstm = Embedding(input_dim=source_voc_size + 1, output_dim=emd_dim, input_length=input_seq_length,
                              mask_zero=True, trainable=trainable_embeddings, weights=[source_words])(word_input)
    word_embedding_cnn = Embedding(input_dim=source_voc_size + 1, output_dim=emd_dim, input_length=input_seq_length,
                                  mask_zero=False, trainable=trainable_embeddings, weights=[source_words])(word_input)

    ent_embedding_lstm = Embedding(input_dim=ent_voc_size + 1, output_dim=ent_voc_size + 1,
                                  input_length=input_seq_length,
                                  mask_zero=True, trainable=True, weights=[ent_voc])(ent_input)
    ent_embedding_cnn = Embedding(input_dim=ent_voc_size + 1, output_dim=ent_voc_size + 1,
                                      input_length=input_seq_length,
                                      mask_zero=False, trainable=True, weights=[ent_voc])(ent_input)

    # Concatenate the input and apply dropout.
    concat_input_lstm = concatenate([word_embedding_lstm, ent_embedding_lstm], axis=-1)
    concat_input_lstm = Dropout(0.2)(concat_input_lstm)
    concat_input_cnn = concatenate([word_embedding_cnn, ent_embedding_cnn], axis=-1)
    concat_input_cnn = Dropout(0.2)(concat_input_cnn)

    # Define BiLSTM and CNN layers.
    lstm = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1))(concat_input_lstm)

    cnn = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_cnn)
    max_pool = GlobalMaxPooling1D()(cnn)
    repeat_max_pool = RepeatVector(input_seq_length)(max_pool)

    # Concatenate BiLSTM and CNN.
    concat_layer = concatenate([lstm, repeat_max_pool], axis=-1)
    concat_layer = Dropout(0.2)(concat_layer)

    # Include time distributed layer.
    time_distributed = TimeDistributed(Dense(target_voc_size + 1))(concat_layer)

    # Define the CRF layer (use 'marginal' to yield probabilities instead of one-hot encoded prediction).
    crf = CRF(target_voc_size + 1, sparse_target=False, learn_mode='marginal', test_mode='marginal')
    all_layers = crf(time_distributed)

    # Finalize and compile the model.
    model = Model([word_input, ent_input], all_layers)
    model.compile(loss=losses.crf_loss, optimizer='adam', metrics=[metrics.crf_accuracy])

    return model


def create_cnn_model(source_voc_size, target_voc_size, ent_voc_size, source_words,
                     ent_voc, input_seq_length, output_seq_length,
                     hidden_dim, emd_dim, trainable_embeddings=False):
    """
    Construct a CNN open relation extraction model.

    :param source_voc_size: The size of the source vocabulary.
    :param target_voc_size: The size of the target vocabulary.
    :param ent_voc_size: The size of the entity tag vocabulary.
    :param source_words: The input word embeddings.
    :param ent_voc: The entity tag vocabulary.
    :param input_seq_length: The maximum input sentence length.
    :param output_seq_length: The output sequence length.
    :param hidden_dim: The dimensionality of the hidden layers.
    :param emd_dim: The dimensionality of the word embeddings.
    :param trainable_embeddings: Option to optimize the word embeddings during training.

    :return: A non-trained open relation extraction model.
    """
    # Define the model input.
    word_input = Input(shape=(input_seq_length,), dtype='int32')
    ent_input = Input(shape=(input_seq_length,), dtype='int32')

    word_embedding = Embedding(input_dim=source_voc_size + 1, output_dim=emd_dim, input_length=input_seq_length,
                                  mask_zero=False, trainable=trainable_embeddings, weights=[source_words])(word_input)

    ent_embedding = Embedding(input_dim=ent_voc_size + 1, output_dim=ent_voc_size + 1,
                                      input_length=input_seq_length,
                                      mask_zero=False, trainable=True, weights=[ent_voc])(ent_input)

    # Concatenate the input and apply dropout.
    concat_input = concatenate([word_embedding, ent_embedding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)

    # Define CNN layers.
    cnn = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input)
    max_pool = GlobalMaxPooling1D()(cnn)
    repeat_max_pool = RepeatVector(input_seq_length)(max_pool)

    # Include time distributed layer.
    time_distributed = TimeDistributed(Dense(target_voc_size + 1))(repeat_max_pool)

    # Define the CRF layer (use 'marginal' to yield probabilities instead of one-hot encoded prediction).
    crf = CRF(target_voc_size + 1, sparse_target=False, learn_mode='marginal', test_mode='marginal')
    all_layers = crf(time_distributed)

    # Finalize and compile the model.
    model = Model([word_input, ent_input], all_layers)
    model.compile(loss=losses.crf_loss, optimizer='adam', metrics=[metrics.crf_accuracy])

    return model


def create_lstm_model(source_voc_size, target_voc_size, ent_voc_size, source_words,
                      ent_voc, input_seq_length, output_seq_length,
                      hidden_dim, emd_dim, trainable_embeddings=False):
    """
    Construct a BiLSTM open relation extraction model.

    :param source_voc_size: The size of the source vocabulary.
    :param target_voc_size: The size of the target vocabulary.
    :param ent_voc_size: The size of the entity tag vocabulary.
    :param source_words: The input word embeddings.
    :param ent_voc: The entity tag vocabulary.
    :param input_seq_length: The maximum input sentence length.
    :param output_seq_length: The output sequence length.
    :param hidden_dim: The dimensionality of the hidden layers.
    :param emd_dim: The dimensionality of the word embeddings.
    :param trainable_embeddings: Option to optimize the word embeddings during training.

    :return: A non-trained open relation extraction model.
    """
    # Define the model input.
    word_input = Input(shape=(input_seq_length,), dtype='int32')
    ent_input = Input(shape=(input_seq_length,), dtype='int32')

    word_embedding = Embedding(input_dim=source_voc_size + 1, output_dim=emd_dim, input_length=input_seq_length,
                              mask_zero=True, trainable=trainable_embeddings, weights=[source_words])(word_input)

    ent_embedding = Embedding(input_dim=ent_voc_size + 1, output_dim=ent_voc_size + 1,
                                  input_length=input_seq_length,
                                  mask_zero=True, trainable=True, weights=[ent_voc])(ent_input)

    # Concatenate the input and apply dropout.
    concat_input = concatenate([word_embedding, ent_embedding], axis=-1)
    concat_input = Dropout(0.2)(concat_input)

    # Define BiLSTM layer and apply dropout.
    lstm = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1))(concat_input)
    dropout_lstm = Dropout(0.2)(lstm)

    # Include time distributed layer.
    time_distributed = TimeDistributed(Dense(target_voc_size + 1))(dropout_lstm)

    # Define the CRF layer (use 'marginal' to yield probabilities instead of one-hot encoded prediction).
    crf = CRF(target_voc_size + 1, sparse_target=False, learn_mode='marginal', test_mode='marginal')
    all_layers = crf(time_distributed)

    # Finalize and compile the model.
    model = Model([word_input, ent_input], all_layers)
    model.compile(loss=losses.crf_loss, optimizer='adam', metrics=[metrics.crf_accuracy])

    return model


def test_model(model, test_data, test_entities, word_index):
    """
    Test the open relation extraction model.

    :param model: The trained model.
    :param test_data: The test sentences and labels.
    :param test_entities: The test entities.
    :param word_index: The indices of input vocabulary.

    :return: The evaluation metrics.
    """
    word_index[0] = ''
    test_sentences = np.asarray(test_data[0], dtype="int32")
    test_labels = np.asarray(test_data[1], dtype="int32")
    test_entities = np.asarray(test_entities, dtype="int32")

    batch_size = 50
    test_length = len(test_sentences)
    test_count = 0
    # Ensure that the test number fits the batch size.
    if len(test_sentences) % batch_size == 0:
        number_of_tests = len(test_sentences) / batch_size
    else:
        extra_test_num = batch_size - len(test_sentences) % batch_size

        extra_data = test_sentences[:extra_test_num]
        test_sentences = np.append(test_sentences, extra_data, axis=0)

        extra_data = test_labels[:extra_test_num]
        test_labels = np.append(test_labels, extra_data, axis=0)

        extra_data = test_entities[:extra_test_num]
        test_entities = np.append(test_entities, extra_data, axis=0)

        number_of_tests = len(test_sentences) / batch_size

    test_result = []

    # Compute the test predictions.
    for n in range(0, int(number_of_tests)):
        print('Test batch: ' + str(n + 1) + ' / ' + str(number_of_tests))
        batch_sentences = test_sentences[n * batch_size:(n + 1) * batch_size]
        batch_entities = test_entities[n * batch_size:(n + 1) * batch_size]
        batch_labels = test_labels[n * batch_size:(n + 1) * batch_size]

        predictions = model.predict([batch_sentences, batch_entities])

        for i in range(0, len(predictions)):
            if test_count < test_length:
                sent = predictions[i]
                prediction = []

                for word in sent:
                    next_index = np.argmax(word)
                    next_token = word_index[next_index]
                    prediction.append(next_token)

                sent_label = batch_labels[i]
                true_label = []
                for word in sent_label:
                    next_token = word_index[word]
                    true_label.append(next_token)

                result = []
                result.append(prediction)
                result.append(true_label)
                test_count += 1
                test_result.append(result)

    # Compute test metrics from predictions.
    P, R, F, PR_count, P_count, TR_count = evaluation_rel(test_result)

    return P, R, F, PR_count, P_count, TR_count


def predict_labels(model, test_data, test_ent_data, word_indices):
    """
    Compute predictions on the test set.

    :param model: The trained model.
    :param test_data: The test data.
    :param test_ent_data: The test entities.
    :param word_indices: The input word indices.

    :return: The predictions.
    """
    word_indices[0] = ''
    test_instances = np.asarray(test_data, dtype="int32")
    test_entities = np.asarray(test_ent_data, dtype="int32")

    batch_size = 50
    # Ensure that the test number fits the batch size.
    if len(test_instances) % batch_size == 0:
        number_of_tests = len(test_instances) / batch_size
    else:
        extra_test_num = batch_size - len(test_instances) % batch_size

        extra_data = test_instances[:extra_test_num]
        test_instances = np.append(test_instances, extra_data, axis=0)

        extra_data = test_entities[:extra_test_num]
        test_entities = np.append(test_entities, extra_data, axis=0)

        number_of_tests = len(test_instances) / batch_size

    test_result = []
    # Compute the test predictions.
    for n in range(0, int(number_of_tests)):
        print('Prediction batch: ' + str(n + 1) + ' / ' + str(number_of_tests))
        batch_instances = test_instances[n * batch_size:(n + 1) * batch_size]
        batch_entities = test_entities[n * batch_size:(n + 1) * batch_size]

        predictions = model.predict([batch_instances, batch_entities])
        test_result.append(predictions)

    return test_result


def select_model(model_type, source_voc_size, target_voc_size, ent_voc_size, source_words,
                 ent_voc, input_seq_length, output_seq_length,
                 hidden_dim, emd_dim, trainable_embeddings=False):
    """
    Construct an open relation extraction model from the given model type.

    :param source_voc_size: The size of the source vocabulary.
    :param target_voc_size: The size of the target vocabulary.
    :param ent_voc_size: The size of the entity tag vocabulary.
    :param source_words: The input word embeddings.
    :param ent_voc: The entity tag vocabulary.
    :param input_seq_length: The maximum input sentence length.
    :param output_seq_length: The output sequence length.
    :param hidden_dim: The dimensionality of the hidden layers.
    :param emd_dim: The dimensionality of the word embeddings.
    :param trainable_embeddings: Option to optimize the word embeddings during training.

    :return: A non-trained open relation extraction model.
    """
    model = None
    if model_type is 'cnn_lstm':
        model = create_cnn_lstm_model(source_voc_size=source_voc_size,
                                      target_voc_size=target_voc_size,
                                      ent_voc_size=ent_voc_size,
                                      source_words=source_words,
                                      ent_voc=ent_voc,
                                      input_seq_length=input_seq_length,
                                      output_seq_length=output_seq_length,
                                      hidden_dim=hidden_dim, emd_dim=emd_dim,
                                      trainable_embeddings=trainable_embeddings)

    elif model_type is 'cnn':
        model = create_cnn_model(source_voc_size=source_voc_size,
                                 target_voc_size=target_voc_size,
                                 ent_voc_size=ent_voc_size,
                                 source_words=source_words,
                                 ent_voc=ent_voc,
                                 input_seq_length=input_seq_length,
                                 output_seq_length=output_seq_length,
                                 hidden_dim=hidden_dim, emd_dim=emd_dim,
                                 trainable_embeddings=trainable_embeddings)

    elif model_type is 'lstm':
        model = create_lstm_model(source_voc_size=source_voc_size,
                                  target_voc_size=target_voc_size,
                                  ent_voc_size=ent_voc_size,
                                  source_words=source_words,
                                  ent_voc=ent_voc,
                                  input_seq_length=input_seq_length,
                                  output_seq_length=output_seq_length,
                                  hidden_dim=hidden_dim, emd_dim=emd_dim,
                                  trainable_embeddings=trainable_embeddings)

    return model


def train_model(model_type, data_file, model_file, batch_size=50, retrain=False, transfer_learning=False,
                transfer_learning_file="", trainable_embeddings=False):
    """
    Train an open relation extraction model.

    :param model_type: The type of the model (cnn, lstm or cnn_lstm).
    :param data_file: The path to the data file.
    :param model_file: The path where the model weights should be stored.
    :param batch_size: The training batch size.
    :param retrain: Boolean which indicates if the model should be retrained.
    :param transfer_learning: Boolean which indicates if transfer learning from an existing model should be employed.
    :param transfer_learning_file: Source model for transfer learning.
    :param trainable_embeddings: Boolean which indicates of the embeddings should be optimized during training.

    :return: A trained open relation extraction model.
    """

    # Load the created training data.
    train_data, test_data, source_words, source_voc, source_index_word, \
    target_voc, target_index_word, max_s, k, \
    train_ent_data, test_ent_data, ent_words, ent_voc, ent_index_word = pickle.load(
        open(data_file, 'rb'))

    # Extract the training data.
    x_train = np.asarray(train_data[0], dtype="int32")
    y_train = np.asarray(train_data[1], dtype="int32")
    ent_train = np.asarray(train_ent_data, dtype="int32")

    # Create the selected model.
    model = select_model(model_type, source_voc_size=len(source_voc), target_voc_size=len(target_voc),
                         ent_voc_size=len(ent_voc),
                         source_words=source_words, ent_voc=ent_words,
                         input_seq_length=max_s,
                         output_seq_length=max_s,
                         hidden_dim=200, emd_dim=k, trainable_embeddings=trainable_embeddings)

    # Apply retraining or transfer learning.
    if retrain:
        model.load_weights(model_file)
    if transfer_learning:
        model.load_weights(transfer_learning_file)

    # Print a summary of the created model.
    model.summary()

    # Reformat the training data to make it suitable for model input.
    train_instances, train_entities, train_labels, val_instances, val_entities, val_labels = get_training_input(x_train,
                                                                        ent_train, y_train,
                                                                        max_s, max_s,
                                                                        len(target_voc),
                                                                        target_index_word,
                                                                        shuffle=True)

    # Define model options.
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint(filepath="./data/model/checkpoint_model.h5", monitor='val_loss', verbose=0,
                                 save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.0001)

    # Train the model on the training data.
    model.fit([train_instances, train_entities], train_labels,
              batch_size=batch_size,
              epochs=100,
              verbose=1,
              shuffle=True,
              # validation_split=0.2,
              validation_data=([val_instances, val_entities], val_labels),
              callbacks=[reduce_lr, checkpoint, early_stopping])

    # Store the model weights.
    save_model(model, model_file)

    return model


def load_model(model_type, data_file, model_file, other_test_language=False, other_test_file='',
               test_language_embedding=''):
    """
    Load a trained open relation extraction model.

    :param model_type: The type of the model (cnn, lstm or cnn_lstm).
    :param data_file: The path to the data file.
    :param model_file: The path where the model weights are stored.
    :param other_test_language: Boolean which indicates if the test language differs from the training language.
    :param other_test_file: The test file should be defined if the test language differs from the training language.
    :param test_language_embedding: Them embedding file should be defined if the test language differs.

    :return: The trained model and information about the model.
    """
    # Load the created training data.
    _, _, source_words, source_voc, _, target_voc, target_index_word, max_s, k, _, _, ent_words, ent_voc, _ = \
        pickle.load(open(data_file, 'rb'))

    if other_test_language:
        # Load the model including word embeddings for the test language.
        model, source_voc = load_model_other_test_language(source_voc, source_words, other_test_file,
                                                           test_language_embedding, model_type, target_voc, ent_voc,
                                                           ent_words, k, model_file)
    else:
        # Load the model assuming that the test language is the same as the training language.
        model = select_model(model_type, source_voc_size=len(source_voc), target_voc_size=len(target_voc),
                             ent_voc_size=len(ent_voc),
                             source_words=source_words, ent_voc=ent_words,
                             input_seq_length=50,
                             output_seq_length=max_s,
                             hidden_dim=200, emd_dim=k)

        model.load_weights(model_file)

    return model, target_index_word, max_s, source_voc, target_voc, ent_voc


def load_model_other_test_language(source_voc, source_words, other_test_file, test_language_embedding, model_type,
                                   target_voc, ent_voc, ent_words, k, model_file):
    """
    Should be used if the test language differs from the training language. This function adds word embeddings for
    the test language to the model.

    :param source_voc: The source vocabulary.
    :param source_words: The source word embeddings.
    :param other_test_file: The test file.
    :param test_language_embedding: The embeddings for the test language.
    :param model_type: The model type (cnn, lstm or cnn_lstm).
    :param target_voc: The target vocabulary.
    :param ent_voc: The entity tag vocabulary.
    :param ent_words: The enitity indices.
    :param k: The embedding dimension.
    :param model_file: The file were the model weights are stored.

    :return: A loaded model and extended source vocabulary.
    """
    # Extend the source vocabulary with words from the test language.
    start_count = len(source_voc) + 1
    additional_source_vob, _, _, _, _ = get_word_index(other_test_file, other_test_file, start_count)

    duplicate_vob = set(source_voc.keys()) & set(additional_source_vob.keys())
    for key in duplicate_vob:
        del additional_source_vob[key]

    _, _, additional_source_W = load_vec_txt(test_language_embedding, additional_source_vob,
                                             embeddings_per_language=300000, start_count=start_count,
                                             unknown_embedding=source_words[source_voc["UNK"]])
    additional_source_W = additional_source_W[:-1]
    source_words = np.vstack((source_words, additional_source_W[start_count:]))
    source_voc.update(additional_source_vob)

    # Create the model.
    model = select_model(model_type, source_voc_size=len(source_voc), target_voc_size=len(target_voc),
                         ent_voc_size=len(ent_voc),
                         source_words=source_words, ent_voc=ent_words,
                         input_seq_length=50,
                         output_seq_length=50,
                         hidden_dim=200, emd_dim=k)

    # Add test language word embeddings to the existing model weights.
    temp_file_path = './data/model/temp_model.h5'
    copyfile(model_file, temp_file_path)
    w_file = h5py.File(temp_file_path, 'r+')

    embedding1 = w_file['embedding_1']['embedding_1']['embeddings:0']
    embedding2 = w_file['embedding_2']['embedding_2']['embeddings:0']

    merged_embedding1 = np.vstack((embedding1[:start_count], source_words[start_count:]))
    merged_embedding2 = np.vstack((embedding2[:start_count], source_words[start_count:]))

    del w_file['embedding_1']['embedding_1']['embeddings:0']
    del w_file['embedding_2']['embedding_2']['embeddings:0']
    w_file['embedding_1']['embedding_1']['embeddings:0'] = merged_embedding1
    w_file['embedding_2']['embedding_2']['embeddings:0'] = merged_embedding2
    w_file.close()

    # Load the model weights and remove the temporary weights file.
    model.load_weights(temp_file_path)
    os.remove(temp_file_path)

    return model, source_voc


if __name__ == "__main__":
    # Define the model type (cnn, lstm or cnn_lstm).
    model_type = 'cnn_lstm'

    # Define the model language.
    language = 'fr'

    print('Model choice: ' + model_type)
    print('Language choice: ' + language)

    # Define the maximum number of word embeddings used.
    embeddings_per_language = 300000

    # Define the path to the word embedding file.
    word_embedding_files = {language: "./data/embeddings/fastText/cc." + language + ".300.vec"}

    # Define the path where the data and model files should be stored.
    data_file = "./data/datafile/" + language + "_fasttext_wmorc_data.pkl"
    model_file = "./data/model/" + language + "_fasttext_wmorc_cnn_lstm_prob.h5"

    # Define the transfer learning source model path.
    transfer_learning_model_file = "./data/model/en_fasttext_wmorc_cnn_lstm_prob.h5"

    # Define the training and test file paths.
    train_file = "../Data-generation/WMORC/wmorc_" + language + "_rel_train.json"
    train_ent_file = "../Data-generation/WMORC/wmorc_" + language + "_ent_train.json"

    test_file = "../Data-generation/WMORC/wmorc_" + language + "_rel_test.json"
    test_ent_file = "../Data-generation/WMORC/wmorc_" + language + "_ent_test.json"

    # Option to output precision, recall and F1-score on the test set.
    test_option = True
    # Option to employ transfer learning from an existing model.
    transfer_learning_option = False
    # Option to retrain the current model.
    retrain_option = False

    if not os.path.exists(data_file):
        print("--++ Creating data file ++--")
        create_data_file(train_file, test_file, word_embedding_files, data_file,
                         train_ent_file, test_ent_file, max_sen_length=50,
                         embeddings_per_language=embeddings_per_language)

    if not os.path.exists(model_file):
        print("--++ Training model ++--")
        print("Data file found: " + data_file)
        if transfer_learning_option:
            print("Transfer weights from: " + transfer_learning_model_file)
            train_model(model_type, data_file, model_file, transfer_learning=transfer_learning_option,
                        transfer_learning_file=transfer_learning_model_file)
        else:
            train_model(model_type, data_file, model_file)
    else:
        if retrain_option:
            print("--++ Retraining model ++--")
            train_model(model_type, data_file, model_file, retrain=retrain_option)

    if test_option:
        print("-- Testing model --")

        model, target_index_word, max_sen_length, source_voc, target_voc, ent_voc = load_model(model_type, data_file,
                                                                                               model_file)

        test_data = create_index_word(test_file, max_sen_length, source_voc, target_voc)
        test_ent_data = create_index_ent(test_ent_file, max_sen_length, ent_voc)

        P, R, F, PR_count, P_count, TR_count = test_model(model, test_data, test_ent_data,
                                                          target_index_word)
        print('P= ', P, '  R= ', R, '  F= ', F)
