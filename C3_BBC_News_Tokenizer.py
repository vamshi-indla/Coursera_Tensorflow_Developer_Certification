# source
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Exercise-answer.ipynb

'''
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv \
    -O /tmp/bbc-text.csv

'''
stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
              "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
              "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
              "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
              "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its",
              "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
              "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
              "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
              "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
              "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've",
              "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
              "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
              "yourself", "yourselves"];

category = []
text = []
with open('/tmp/bbc-text.csv') as f:
    first = 1
    for line in f.readlines():
        if first == 1:
            first = 0
        else:
            category.append(line.split(',')[0])
            sentence = line.split(',')[1]
            for word in sentence.split(' '):
                if word in stop_words:
                    sentence = sentence.replace(' ' + word + ' ', ' ')
            text.append(sentence)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=100000, oov_token='oov')
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

padded = pad_sequences(sequences, padding='post', maxlen=40)
print(padded.shape)

# ==================

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(category)
label_seq = label_tokenizer.texts_to_sequences(category)
label_word_index = label_tokenizer.word_index
print(label_seq)
print(label_word_index)
