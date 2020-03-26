import pandas as pd
import numpy as np
import re
import pretty_midi

from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Input,  Embedding, dot, Activation, concatenate, Flatten
from keras import backend as K
from keras.layers import LSTM
from keras import optimizers
from gensim.models import KeyedVectors

"""Setting parameters for the model"""
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 10
VALIDATION_SPLIT=.2


def build_model(word_index, embedding_weights):
    """
    :param word_index: vocabulary of the odel
    :param embedding_weights:
    :return compiled model
    bui
    """
    lr = 1e-4
    dec_units = enc_units = 100

    en_shape = (221, 140)

    dropout_keep_prob = 0.2
    max_sequence_length = MAX_SEQUENCE_LENGTH = 10
    embedding_size = 300
    vocabulary_size = len(word_index) + 1
    lr = 1e-4

    K.clear_session()
    encoder_input = Input(shape=(en_shape), name='encoder_input')

    encoder = LSTM(enc_units,
                   return_sequences=True,
                   recurrent_initializer='glorot_uniform',
                   return_state=True,
                   name='enc_lstm')(encoder_input)

    encoder_outputs, state_h, state_c = encoder
    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=(max_sequence_length,), name='decoder_input')
    x = Embedding(vocabulary_size,
                  embedding_size,
                  weights=[embedding_weights],
                  input_length=max_sequence_length,
                  trainable=False,
                  name='embedding')(decoder_input)
    decoder_output = LSTM(units=dec_units,
                          return_sequences=True,
                          name='dec_lstm')(x, initial_state=encoder_states)

    attention = dot([decoder_output, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax')(attention)
    context = dot([attention, encoder_outputs], axes=[2, 1])
    decoder_combined_context = concatenate([context, decoder_output])
    x = Flatten()(decoder_combined_context)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_keep_prob)(x)
    output = Dense(vocabulary_size, activation="softmax")(x)
    model = Model(inputs=[encoder_input, decoder_input], outputs=[output])

    adam = optimizers.Adam(lr=lr)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


def clean_text(text):
    """
    :param text:
    :return: Clean text without special characters, change '&' to 'eos'
    """
    text=text.replace('&',' eos ')
    text=text.lower()
    text=re.sub(r'\[.*?\]', '', text)   #remove remarks in the text
    text=re.sub("[^\na-z 0-9]", "",text).strip()
    text=re.sub(' +', ' ', text)
    return text


def create_midis_vector(df):
    """
    input - DataFrame - where first and second columns are the artist&song names
    output- midis_vector - list of tuples in the form: (song#, pretty_midi_object)
    """
    midis = []
    for row in df.itertuples():
        name = (str(row[1])+" - "+str(row[2])).replace(' ','_') + '.mid'
        try:
            midis.append((row[0],pretty_midi.PrettyMIDI("midi_files/"+name)))
        except:
            pass
    return midis


def test_create_midis_vector(df):
    """
    input - DataFrame - where first and second columns are the artist&song names
    output- midis_vector - list of tuples in the form: (song#, pretty_midi_object)
    """
    midis = []
    for row in df.itertuples():
        name = (str(row[1])+" -"+str(row[2])).replace(' ','_') + '.mid'
        try:
            midis.append((row[0],pretty_midi.PrettyMIDI("midi_files/"+name)))
        except:
            pass
    return midis


def get_median_length(midis_vector):
    """
    Input: midis_vector - vector that includes the pretty midi objects
    Output: Median length of all songs
    """
    songs_time = []
    for midi in midis_vector:
        songs_time.append(midi[1].get_end_time())
    return int(np.median(songs_time))


def create_midi_matrix(midi_file, number_of_sequences):
    """
    Input:
    midi file to extract features
    number of sequence (median of songs length) - for the RNN input dimensions
    Output: matrix for the RNN input with dimensions- (number of sequences X 140)
    """
    piano = midi_file.get_piano_roll(fs=1).T
    chroma = midi_file.get_chroma(fs=1).T
    matrix = np.concatenate((chroma,piano),axis=1)
    if len(matrix) >= number_of_sequences: # Midi vec is larger, take the middle part
        diff = len(matrix) - number_of_sequences
        cut_from_start = int(diff/2)
        cut_from_end = int(diff/2 + diff%2)
        matrix = matrix[cut_from_start:len(matrix)-cut_from_end,:]
    else: # Midi vector is smaller than the require input, pad
        diff = number_of_sequences - len(matrix)
        pad_start = np.zeros((int(diff/2),140))
        pad_end = np.zeros((int(diff/2+diff%2),140))
        matrix = np.concatenate((pad_start,matrix,pad_end),axis=0)
    return matrix


def concat_(text):
    return text.replace(' ','_')


def clean_singer_song(singer,song_name):
    """
    :param singer:
    :param song_name:
    :return: the singer_song name to retrieve files
    """
    singer_song=concat_(clean_text(singer+' '+song_name))
    return singer_song


def create_training_data(song_index, songs_dict, X, Y):
    """
    :param song_index: index of all songs in training data
    :param songs_dict: midi matrices for all songs
    :param X: input sequences matrix
    :param Y: output words vector
    :return: x_train, midi_train, y_train, x_test, midi_test, y_test
    """
    midi_list = []
    x_train_list = []
    y_train_list = []

    midi_test_list = []
    x_test_list = []
    y_test_list = []

    last_index = np.unique(song_index)[-5:]
    for i, ind in enumerate(song_index):
        if ind not in last_index:
            midi_list.append(songs_dict[ind])
            x_train_list.append(X[i])
            y_train_list.append(Y[i])
        else:
            midi_test_list.append(songs_dict[ind])
            x_test_list.append(X[i])
            y_test_list.append(Y[i])
    return x_train_list, midi_list, y_train_list, x_test_list, midi_test_list, y_test_list


def load_model(name):
    """
    :param name: name of the loaded model
    :return: trained model
    """
    # Model reconstruction from JSON file
    with open('models/'+name+'_'+'model_architecture.json', 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights('models/'+name+'_'+'model_weights.h5')
    return model


def save_model(model,name):
    """
    :param model: Trained so-far model
    :param name: name for saving the model
    """
    # Save the weights
    model.save_weights('models/'+name+'_'+'model_weights.h5')

    # Save the model architecture
    with open('models/'+name+'_'+'model_architecture.json', 'w') as f:
        f.write(model.to_json())


def shuffle_list(x_train_list, midi_list, y_train_list):
    """
    :param x_train_list:
    :param midi_list:
    :param y_train_list:
    :return: shuffling the trianing data
    """
    order_ind = np.random.permutation(np.arange(len(y_train_list)))
    x_train_list = [x_train_list[i] for i in order_ind]
    midi_list = [midi_list[i] for i in order_ind]
    y_train_list = [y_train_list[i] for i in order_ind]
    return x_train_list, midi_list, y_train_list


def gendata(x_train_list, midi_list, y_train_list, batch_size):
    """
    :param x_train_list:
    :param midi_list:
    :param y_train_list:
    :param batch_size:
    :return: combining midi matrix and corresponding sequences
    """
    while True:
        x_train_list, midi_list, y_train_list = shuffle_list(x_train_list, midi_list, y_train_list)
        num_batches = int(len(y_train_list) / batch_size)
        for ii in range(0, num_batches):
            cur_batch_idx = ii * batch_size
            if cur_batch_idx + batch_size > len(y_train_list):
                batch_x = np.array(x_train_list[cur_batch_idx:])
                batch_midi = np.array(midi_list[cur_batch_idx:])
                batch_y = np.array(y_train_list[cur_batch_idx:])
            else:
                batch_x = np.array(x_train_list[cur_batch_idx:cur_batch_idx + batch_size])
                batch_midi = np.array(midi_list[cur_batch_idx:cur_batch_idx + batch_size])
                batch_y = np.array(y_train_list[cur_batch_idx:cur_batch_idx + batch_size])

            yield [batch_midi, batch_x], batch_y


def normalize(probs):
    """
    :param probs: list of probabilities
    :return: normalized probabilities
    """
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]


def get_next_word(model, input_sequence, input_melody, random_state):
    """
    :param model: trained model
    :param input_sequence: seqeuence input
    :param input_melody: melody vector
    :param random_state: choose the next word from top int(random_state) options
    :return: randomly chosen word based on probabilities and random state
    """
    predicted_word = model.predict([input_melody, np.array(input_sequence)])
    top_next_words = (-predicted_word[0]).argsort()[:random_state]
    top_prediced_probs = predicted_word[0][top_next_words]
    normalized_top_words = normalize(top_prediced_probs)
    # next_word = top_next_words[np.random.randint(randomn_state)]
    return np.random.choice(top_next_words, 1, p=normalized_top_words)[0]


def create_song(model, initial_word, input_melody, lyrics_length=50, random_state=1,num_of_songs=3):
    """
    :param model: trained model
    :param input_sequence: seqeuence input
    :param input_melody: melody vector
    :param lyrics_length: the length of the song
    :param random_state: choose the next word from top int(random_state) options
    :param num_of_songs: number of different songs with the same melody
    :return:
    """
    predicted_sentence = []
    for j in range(num_of_songs):
        predicted_sentence.append([initial_word])
        input_sequence = [np.zeros(10)]
        input_sequence[0][9] = initial_word
        for ii in range(lyrics_length-1):
            next_word = get_next_word(model, input_sequence, input_melody, random_state)
            input_sequence = np.roll(input_sequence,9)
            input_sequence[0][9] = next_word
            predicted_sentence[j].append(next_word)
    return np.array(predicted_sentence)


