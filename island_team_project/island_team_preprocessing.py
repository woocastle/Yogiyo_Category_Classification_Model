# preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle

pd.set_option('display.unicode.east_asian_width', True)                 # 줄 맞춰주기
df = pd.read_csv('./crawling_data.csv')
print(df.head())
print(df.category.value_counts())

X = df['restaurant_name']
Y = df['category']

encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)        # 카테고리를 숫자로 라벨링!
print(labeled_Y[:5])
print(encoder.classes_)     # 몇 번으로 라벨링 되어있는지 확인 할수 있다.(보통 오름차순으로 정렬을 해서 받는다)
with open('./label_encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)
onehot_Y = to_categorical(labeled_Y)
print(onehot_Y[:5])

token = Tokenizer()     # 자연어 처리를 위해 라벨링, 문장으로부터 단어를 토큰화하고 숫자에 대응시키는 딕셔너리를 사용
token.fit_on_texts(X)
tokened_X = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1
with open('./yogiyo_token.pickle', 'wb') as f:
    pickle.dump(token, f)

# 문장 길이 맞추기!
max_len = 0
for i in range(len(tokened_X)):
    if max_len < len(tokened_X[i]):
        max_len = len(tokened_X[i])
print(max_len)

X_pad = pad_sequences(tokened_X, max_len)

# 테스트 데이터 나누기
X_train, X_test, Y_train, Y_test = train_test_split(
    X_pad, onehot_Y, test_size=0.3)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# 데이터 저장히기
xy = X_train, X_test, Y_train, Y_test
np.save('./yogiyo_data_max_{}_wordsize_{}.npy'.format(max_len, wordsize), xy)