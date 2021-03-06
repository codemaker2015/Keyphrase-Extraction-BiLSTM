import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
import os
from datetime import datetime

import silence_tensorflow.auto
from tensorflow.python.keras.models import Model, Sequential, model_from_json, load_model
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
stop_words = set(stopwords.words('english'))


with open('model/model.json') as json_file:
    json_config = json_file.read()
model = model_from_json(json_config)

# Load weights
model.load_weights('model/model.h5')

tokenizer = Tokenizer()
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

try:
    while True:
        final_results = []
        input_ = input("Enter the file name:")
        final_results.append('\nTimestamp: ' + str(datetime.now()) + '\nFile name: ' + input_ + '\nKeywords: ')
        with open(input_) as fil:
	        lines = fil.readlines()
	        for line in lines:
                    input_ = line
                    new_t = Tokenizer()
                    new_t.fit_on_texts([input_])
                    tokens = [i for i in new_t.word_index.keys()]
                    actual_tokens = new_t.texts_to_sequences([input_])
                    inv_map_tokens = {v: k for k, v in new_t.word_index.items()}
                    actual_tokens = [inv_map_tokens[i] for i in actual_tokens[0]]
                    tokens = actual_tokens
                    input_ = tokenizer.texts_to_sequences([input_])
                    input_ = pad_sequences(input_, padding = "post", truncating = "post", maxlen = 25, value = 0)
                    output = model.predict([input_])
                    output = np.argmax(output, axis = -1)
                    where_ = np.where(output[0] == 1)[0]
                    output_keywords = np.take(tokens, where_)
                    output_keywords = [i for i in output_keywords if i not in stop_words]
                    output_keywords = list(set(output_keywords))
                    final_results.append(', '.join(output_keywords))
        
        print("Keywords: " + str(final_results))
        # Storing the output
        with open('final_results.csv', 'a') as fil:
            fil.writelines(final_results)

except KeyboardInterrupt:
    pass
