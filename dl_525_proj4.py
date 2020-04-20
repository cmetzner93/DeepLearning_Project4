import sys
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import LSTM, SimpleRNN, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from pickle import dump, load
import matplotlib.pyplot as plt
from keras import optimizers

# task 1: function which given a text file name, a window size and a stride creates the training data
def get_train_data(textfile, window_size, stride):
    # open the file as read, read text, and close file
    file = open(textfile, 'r')
    text = file.read()
    file.close()

    # strip of the new line characters so that we
    # have one long sequence of characters separated only
    # by white space
    tokens = text.split()
    data = ' '.join(tokens)

    # get sequences of characters of length window_size+1
    sequences = []
    for i in range(window_size, len(data), stride):
        sequence = data[i - window_size: i + 1]
        sequences.append(sequence)

    # save sequences
    data = '\n'.join(sequences)
    file = open('train_data.txt', 'w')
    file.write(data)
    file.close()


# task 2: function which given a file name in which each line is a single training sequence
# returns the input X and output y array for training
# one-hot encoding is also performed here
def preprocess_data(train_textfile):
    # open the file as read, read text, and close file
    file = open(train_textfile, 'r')
    text = file.read()
    file.close()

    # get list of sequences by splitting the text by new line
    lines = text.split('\n')

    # get unique characters
    chars = sorted(list(set(text)))
    # get mapping of character to integer values and store in a dictionary
    char_to_i_mapping = dict((c, i) for i, c in enumerate(chars))
    # save the mapping
    dump(char_to_i_mapping, open('mapping.pkl', 'wb'))
    # get vocabulary size
    vocab_size = len(char_to_i_mapping)

    # integer encode each sequence of characters using the dictionary mapping
    sequences = []
    for line in lines:
        # integer encode line
        encoded_seq = [char_to_i_mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)

    # now separate the integer encoded sequences into input and output
    sequences = np.array(sequences)
    X = sequences[:, :-1]
    y = sequences[:, -1]

    # now one-hot encode each character, meaning each character becomes a vector of length vocab_size with a 1 marked
    # for the character and 0s elsewhere
    sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
    X = np.array(sequences)
    y = to_categorical(y, num_classes=vocab_size)
    print('X shape: %s and y shape: %s' % (X.shape, y.shape))

    return (X, y)


# task 3: function which predicts a given number of characters given a model, mapping (dictionary with character to
# integer values), window size, initial characters and number of characters to predict
def predict_characters(model, mapping, window_size, init_chars, n_chars):
    text = init_chars
    # predict a fixed number of characters
    for i in range(n_chars):
        # integer encode the characters
        encoded = [mapping[chara] for chara in text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=window_size, truncating='pre')
        # one hot encode
        encoded = to_categorical(encoded, num_classes=len(mapping))
        # predict the next character
        pred_char = model.predict_classes(encoded, verbose=0)
        # reverse mapping of predicted character (integer to character)
        out_char = ''
        for char, index in mapping.items():
            if index == pred_char:
                out_char = char
                break
        # append to input
        text += out_char
    return text


# class used by train_model() function to generate a few sequences by initializing them with random samples from
# the training data and generating the next 10 characters
class predict_during_training(keras.callbacks.Callback):
    def __init__(self, model, sequences):
        self.model = model
        self.sequences = sequences

    def on_epoch_end(self, epoch, logs=None):
        mapping = load(open('mapping.pkl', 'rb'))
        window_size = len(self.sequences[0])
        if epoch % 5 == 0:
            for i in self.sequences:
                text = predict_characters(self.model, mapping, window_size=window_size,
                                          init_chars=i, n_chars=10)
                print(text)  # return prediction
            print()


# task 4: function that trains a specific model, given the model, training data, number of epochs, learning rate
# and model name (needed for file name creation)
def train_model(model, X, y, n_epochs, learning_rate, model_name):
    # open the file as read, read text, and close file
    file = open('train_data.txt', 'r')
    text = file.read()
    file.close()
    # get list of sequences by splitting the text by new line
    lines = text.split('\n')

    # get list of 3 random sequences from training data which will be used to
    # generate/predict characters during training
    random_indexes = list(np.random.randint(low=0, high=len(lines) - X.shape[1] - 1, size=3))
    random_sequences = [lines[index][:-1] for index in random_indexes]

    # compile model
    adam = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    # fit model
    history = model.fit(X, y, epochs=n_epochs, verbose=1,
                        callbacks=[predict_during_training(model, random_sequences)])
    # save model for later use
    model.save('models/%s.h5' % (model_name))
    # save model history
    with open('train_history/%s.pkl' % (model_name), 'wb') as file:
        dump(history.history, file)

    # Plot training loss values vs epochs
    epoch_loss_plot(history.history, model_name)

    return history


# function that creates epoch vs. loss plots
def epoch_loss_plot(history_dict, model_name):
    plt.figure(figsize=(10,8))
    plt.plot(history_dict['loss'])
    plt.title('Loss vs. Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.savefig('plots/epoch_loss_%s.png' %(model_name))


# Driver code main()
def main(argv=None):
    file = argv[1]
    hidden_state = int(argv[3])
    window_size = int(argv[4])
    stride = int(argv[5])

    # create train data
    get_train_data(file, window_size=window_size, stride=stride)
    X, y = preprocess_data('train_data.txt')

    # load the mapping and get vocab size
    mapping = load(open('mapping.pkl', 'rb'))
    vocab_size = len(mapping)

    # build and train rnn
    if argv[2] == 'rnn':
        # build model
        model = Sequential()
        model.add(SimpleRNN(hidden_state, input_dim=vocab_size))
        model.add(Dense(vocab_size, activation='softmax'))
        print(model.summary())

        # train model
        train_model(model, X, y, n_epochs=50, learning_rate=0.001,
                    model_name='rnn_%d_%d_%d' % (int(argv[3]), int(argv[4]), int(argv[5])))

    # build and train multilayer rnn
    elif argv[2] == 'rnn_multi':
        # build model
        model = Sequential()
        model.add(SimpleRNN(hidden_state, input_dim=vocab_size, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(hidden_state))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        print(model.summary())

        # train model
        train_model(model, X, y, n_epochs=50, learning_rate=0.001,
                    model_name='rnn_multi_%d_%d_%d' % (int(argv[3]), int(argv[4]), int(argv[5])))

    # build and train lstm
    elif argv[2] == 'lstm':
        # build model
        model = Sequential()
        model.add(LSTM(hidden_state, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(vocab_size, activation='softmax'))
        print(model.summary())

        # train model
        train_model(model, X, y, n_epochs=50, learning_rate=0.001,
                    model_name='lstm_%d_%d_%d' % (int(argv[3]), int(argv[4]), int(argv[5])))

    # build and train multilayer rnn
    elif argv[2] == 'lstm_multi':
        # build model
        model = Sequential()
        model.add(LSTM(hidden_state, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(hidden_state))
        model.add(Dropout(0.2))
        model.add(Dense(vocab_size, activation='softmax'))
        print(model.summary())

        # train model
        train_model(model, X, y, n_epochs=50, learning_rate=0.001,
                    model_name='lstm_multi_%d_%d_%d' % (int(argv[3]), int(argv[4]), int(argv[5])))


if __name__ == '__main__':
    main(sys.argv)