import pandas as pd
import numpy as np
import re
import pretty_midi

from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Input, Concatenate , Embedding
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
    max_sequence_length = MAX_SEQUENCE_LENGTH
    embedding_size = 300
    vocabulary_size = len(word_index) + 1
    dropout_keep_prob = 0.2
    lr = 1e-4
    lstm_units = 100
    melody_vec_size = 297

    text_input = Input(shape=(max_sequence_length,), name='text_input')
    # melody_input = Input(shape=(1,melody_vec_size,),name='melody_input')
    melody_input = Input(shape=(melody_vec_size,), name='melody_input')

    x = Embedding(vocabulary_size,
                  embedding_size,
                  weights=[embedding_weights],
                  input_length=max_sequence_length,
                  trainable=False,
                  name='embedding')(text_input)
    x = LSTM(units=lstm_units, return_sequences=True)(x)
    x = LSTM(units=lstm_units)(x)
    x = Concatenate(axis=-1)([melody_input, x])
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout_keep_prob)(x)
    preds = Dense(vocabulary_size, activation='softmax')(x)

    model = Model([text_input, melody_input], preds)

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


def get_song_vector(midi_data):
    """
    :param midi_data: pretty_midi object
    :return: vector of summarized data of the pretty midi object
    """
    semitone = midi_data.get_chroma().sum(axis=1) / midi_data.get_chroma().sum()  # 12
    piano_roll = midi_data.get_piano_roll().sum(axis=1) / midi_data.get_piano_roll().sum()  # 128
    transition_matrix = midi_data.get_pitch_class_transition_matrix(normalize=True).flatten()  # 144
    histogram = midi_data.get_pitch_class_histogram()  # 12
    bpm_norm = np.array([midi_data.estimate_tempo() / 300])  # 1

    full_vector = np.concatenate((semitone, piano_roll, transition_matrix, histogram, bpm_norm))
    full_vector[np.isnan(full_vector)] = 0
    return full_vector


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


def create_training_data(song_index, midi_data, X, Y):
    """
    :param song_index: index of all songs in training data
    :param midi_data: midi vectors for all songs
    :param X: input sequences matrix
    :param Y: output words vector
    :return: x_train, midi_train, y_train, x_test, midi_test, y_test
    """
    midi_list=[]
    x_train_list=[]
    y_train_list=[]

    midi_test_list=[]
    x_test_list=[]
    y_test_list=[]

    last_index=np.unique(song_index)[-5:]
    for i,ind in enumerate(song_index):
        if ind not in last_index:
            midi_list.append(midi_data[ind])
            x_train_list.append(X[i])
            y_train_list.append(Y[i])
        else:
            midi_test_list.append(midi_data[ind])
            x_test_list.append(X[i])
            y_test_list.append(Y[i])
    #%%
    x_train=np.array(x_train_list)
    midi_train=np.array(midi_list)
    y_train=np.array(y_train_list)

    x_test=np.array(x_test_list)
    midi_test=np.array(midi_test_list)
    y_test=np.array(y_test_list)
    return x_train, midi_train, y_train, x_test, midi_test, y_test


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
    predicted_word = model.predict([input_sequence, input_melody.reshape(1, -1)])
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


