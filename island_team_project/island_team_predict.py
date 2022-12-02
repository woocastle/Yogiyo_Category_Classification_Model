import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle
from keras.models import load_model

pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', 15)
df = pd.read_csv('./crawling_data_test.csv')
print(df.head())
df.info()

X = df['restaurant_name']
Y = df['category']

with open('./label_encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)
labeled_Y = encoder.transform(Y)
onehot_Y = to_categorical(labeled_Y)

with open('./yogiyo_token.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 5:
        tokened_X[i] = tokened_X[i][:5] # 20개까지만 슬라이싱
X_pad = pad_sequences(tokened_X, 5)

model = load_model('./yogiyo_category_classification_model_0.734.h5')
preds = model.predict(X_pad)
label = encoder.classes_ # label은 label인코더가 가지고 있다.
category_preds = []
for pred in preds:
    category_pred = label[np.argmax(pred)]
    category_preds.append(category_pred)
df['predict'] = category_preds

print(df.head(30))

df['OX'] = False
for i in range(len(df)):
    if df.loc[i, 'category'] == df.loc[i, 'predict']:
        df.loc[i, 'OX'] = True
df.info()

print(df.head(30))
print(df['OX'].value_counts())
print(df['OX'].mean())
print(df.loc[df['OX']==False])
# df.to_csv('./island_data_test_OX.csv', index=False)