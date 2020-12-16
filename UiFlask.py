from keras.preprocessing import sequence
import re
import numpy
import pickle
from io import open
from tensorflow import keras
from flask import Flask, render_template, request
import sys

max_sentence_length = 50
#embedding_vector_length = 300
#dropout = 0.5


app = Flask(__name__)


@app.route('/home',)
def man():
    return render_template('home.html')


@app.route('/home', methods=['POST'])
def home():
    with open("data.pkl", "rb") as fp:  # Unpickling
        vocab_to_int, int_to_vocab, languages_to_int, int_to_languages, languages = pickle.load(fp)

    print("Vocab loaded!")

    # Doc model tu file
    model = load_model()
    print("Model loaded!")
    predSentence = request.form['input']
    result = predict_sentence(model, predSentence, vocab_to_int, int_to_languages)
    print(result)
    return render_template('home.html', result=result)


def process_sentence(sentence):
    # Loai bo cac ki tu dac biet, chuyen cau ve lower case
    return re.sub(r'[\\\\/:*«`\'?¿";!<>,.|]', '', sentence.lower().strip())


def convert_to_int(data, data_int):
    # Chuyen doi text thanh vector so
    all_items = []
    for sentence in data:
        all_items.append([data_int[word] if word in data_int else data_int["<UNK>"] for word in sentence.split()])

    return all_items


def predict_sentence(model, sentence,  vocab_to_int, int_to_languages):
    sentence = process_sentence(sentence)

    # Transform and pad it before using the model to predict
    x = numpy.array(convert_to_int([sentence], vocab_to_int))
    x = sequence.pad_sequences(x, maxlen=max_sentence_length)
    prediction = model.predict(x)

    # Get the highest prediction
    lang_index = numpy.argmax(prediction)
    print(prediction[0][lang_index])

    # Neu probality <0.3 thi hien thi ngon ngu Khong xac dinh/Unknown
    if prediction[0][lang_index]<0.3:
        return "Unknown"
    else:
        return int_to_languages[lang_index]

def load_model():
    model = keras.models.load_model(sys.argv[1])
    return model

if __name__ == "__main__":
    app.run()
