import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding
from tensorflow.keras.layers import TimeDistributed


def convert_prediction_to_equation(prediction, word_index):
    index_to_word = {i: w for w, i in word_index.items()}
    predicted_indices = np.argmax(prediction, axis=-1)
    predicted_words = [index_to_word.get(i, '') for i in predicted_indices.flatten()]
    equation_str = ' '.join(predicted_words).strip()
    equation_str = equation_str.replace('x', 'sp.Symbol("x")')
    equation = eval(equation_str)
    return equation


def preprocess_input(texts):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_seq = pad_sequences(sequences, maxlen=100, padding='post')
    return padded_seq, tokenizer


def preprocess_output(equations, tokenizer):
    sequences = tokenizer.texts_to_sequences(equations)
    padded_seq = pad_sequences(sequences, maxlen=100, padding='post')
    return padded_seq


def load_training_data(filepath):
    with open(filepath, 'r') as file:
        raw_data = file.readlines()

    problems1 = []
    equations1 = []
    problems2= []
    equations2= []
    count=0
    for line in raw_data:
        # Skip empty lines
        if count==200:
            break
        if not line.strip():
            continue

        try:
            count+=1
            problem, equation = line.strip().split(' ||| ')
            if count>200:
                problems2.append(str(problem))
                equations2.append(str(equation))
            else:
                problems1.append(str(problem))
                equations1.append(str(equation))
        except ValueError as e:
            print(f"Error while processing line: {line.strip()}")
            raise e

    X_train, tokenizer = preprocess_input(problems1)
    y_train = preprocess_output(equations1, tokenizer)
    x_test, _ = preprocess_input(problems2)
    y_test = preprocess_output(equations2, tokenizer)

    return X_train, y_train, x_test, y_test, tokenizer


def tokenize_and_pad(texts, maxlen=100):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')
    return padded_sequences, tokenizer


def define_brt_model(vocab_size):
    # Define the BRT model
    model = Sequential([
        Embedding(vocab_size, 64, input_length=100),
        Bidirectional(LSTM(64, return_sequences=True)),
        TimeDistributed(Dense(64, activation='relu')),
        TimeDistributed(Dense(vocab_size, activation='softmax'))
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


X_train, y_train, x_test, y_test, tokenizer = load_training_data('dataset.txt')
vocab_size = len(tokenizer.word_index) + 1
model = define_brt_model(vocab_size)
history = model.fit(X_train, y_train, epochs=10,validation_split=0.1)


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()