import numpy as np 
import pandas as pd 
import tensorflow as tf  
from tensorflow import keras
from keras.layers import TextVectorization  
from keras.layers import Hashing
from keras.layers.experimental.preprocessing import Normalization 
from keras.models import Sequential
from keras.layers import GRU, Dense, Embedding, Dropout

#for refrence, used columns are now: Regular Word (str), Word Length (int), 
# Total Number of Vowels(int), Vowel Constant Pattern (str) 
# with a target of Syllables (int) 
hyp_data = pd.read_pickle('hyp_data.pkl') 
#drop all the columns in combined since theyre useless now 
hyp_data = hyp_data.drop(["Original Raw"], axis = 1) 

''' PREPROCESS, SPLIT INTO TRAINING TEST AND VALIDATION'''

#split to train test and val dataframes
train, val, test = np.split(hyp_data.sample(frac=1), [int(0.8*len(hyp_data)), int(0.9*len(hyp_data))]) 

# Separate training set into features and target
train_features = train.drop('Syllables', axis=1)
train_target = train['Syllables']

# Separate validation set into features and target
val_features = val.drop('Syllables', axis=1)
val_target = val['Syllables']

# Separate test set into features and target
test_features = test.drop('Syllables', axis=1)
test_target = test['Syllables']  

'''FEATURES PREPROCESSING''' 

#text vectorization for the regular word
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    # Keep alphabetic characters, hyphens, apostrophes, and spaces
    return tf.strings.regex_replace(stripped_html, "[^a-zA-Z'-]", "")

text_vectorization = TextVectorization(
    output_mode='int',
    split='character',  
    standardize=custom_standardization, 
    output_sequence_length=50
)

text_vectorization.adapt(train_features['Regular Word'].values)   
vectorized_words_train = tf.cast(text_vectorization(train_features['Regular Word'].values), dtype=tf.float32)
vectorized_words_test = tf.cast(text_vectorization(test_features['Regular Word'].values) , dtype=tf.float32)
vectorized_words_val = tf.cast(text_vectorization(val_features['Regular Word'].values) , dtype=tf.float32)

#normalize number features
#put numeric features together, idk if this is efficent but whatever ill fix l8r
numeric_feature_names = ['Word Length', 'Total Number of Vowels'] 
numeric_feature_train = train_features[numeric_feature_names]
numeric_feature_test = test_features[numeric_feature_names]
numeric_feature_val = val_features[numeric_feature_names]

normalizer = tf.keras.layers.Normalization(axis=-1) 
normalizer.adapt(numeric_feature_train)   

normalized_train_features = normalizer(numeric_feature_train) 
normalized_test_features = normalizer(numeric_feature_test) 
normalized_val_features = normalizer(numeric_feature_val) 

#now we have to preprocess the vc pattern 
#use a hashing function to do this 
#unique_patterns = hyp_data['Vowel Constonant Pattern'].unique()
#num_bins = len(unique_patterns)
#27,114 unique entries, lets set the number of bins to 30,000 just in case  
hashing_layer = Hashing(num_bins=30000)   
hashed_pattern_train_int = tf.expand_dims(hashing_layer(train_features['Vowel Constonant Pattern'].values), axis=-1)
hashed_pattern_test_int = tf.expand_dims(hashing_layer(test_features['Vowel Constonant Pattern'].values), axis=-1 )
hashed_pattern_val_int = tf.expand_dims(hashing_layer(val_features['Vowel Constonant Pattern'].values), axis=-1 )
#convert the int hash into floating point so we can concatenate them later 
hashed_pattern_train = tf.cast(hashed_pattern_train_int, dtype=tf.float32)
hashed_pattern_test = tf.cast(hashed_pattern_test_int, dtype=tf.float32)
hashed_pattern_val = tf.cast(hashed_pattern_val_int, dtype=tf.float32)

'''SLAP EM TOGETHER INTO A DATASET'''
concatenated_train = tf.keras.layers.concatenate([vectorized_words_train, normalized_train_features, hashed_pattern_train], axis=-1) 
concatenated_test = tf.keras.layers.concatenate([vectorized_words_test, normalized_test_features, hashed_pattern_test], axis=-1)  
concatenated_val = tf.keras.layers.concatenate([vectorized_words_val, normalized_val_features, hashed_pattern_val], axis=-1) 
#convert the targets into tensors 
train_labels_tensor = tf.convert_to_tensor(train_target) 
test_labels_tensor = tf.convert_to_tensor(test_target) 
val_labels_tensor = tf.convert_to_tensor(val_target)  

#make da dataset 
batch_size = 32 
train_dataset = (tf.data.Dataset.from_tensor_slices((concatenated_train, train_labels_tensor))
.shuffle(buffer_size=10000)
.cache()  
.batch(batch_size)  
.prefetch(tf.data.AUTOTUNE))

val_dataset = (tf.data.Dataset.from_tensor_slices((concatenated_val, val_labels_tensor))
.cache()  
.batch(batch_size)   
.prefetch(tf.data.AUTOTUNE)) 

test_dataset = (tf.data.Dataset.from_tensor_slices((concatenated_test, test_labels_tensor))
.cache()  
.batch(batch_size)  
.prefetch(tf.data.AUTOTUNE)) 

'''CREATING DA MODEL''' 
#making a seuqential model with a GRU layer 

# Model configuration
max_features = 30000  # Number of unique syllable patterns in the dataset
embedding_dim = 128  # Dimensionality of the embedding layer
gru_units = 128      # Number of units in the GRU layer
num_classes = 20     #syllable counts range from 1-20
dropout_rate = 0.5   # Dropout rate for regularization

model = Sequential() 
model.add(Embedding(input_dim=max_features, output_dim=embedding_dim))
model.add(GRU(units=gru_units, return_sequences=True))
model.add(GRU(units=gru_units))
model.add(Dropout(dropout_rate))
model.add(Dense(num_classes, activation='softmax'))  

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#print(model.summary())  
#train her
history = model.fit(
    train_dataset,
    epochs=1,  
    validation_data=val_dataset
)
print(history.history.keys()) 

