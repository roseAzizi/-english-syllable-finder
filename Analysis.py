import numpy as np 
import pandas as pd 
import matplotlib as plt   
import tensorflow as tf  
from tensorflow import keras
from keras.layers import TextVectorization 
from sklearn.model_selection import train_test_split

hyp_data = pd.read_pickle('hyp_data.pkl') 
#drop all the columns in combined since theyre useless now 
hyp_data = hyp_data.drop(["Original Raw", "Regular Word", "Vowel Constonant Pattern"], axis = 1) 

x_var = hyp_data.drop("Syllables", axis = 1)
y_var = hyp_data [['Syllables']]  
x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size = 0.2) 

#text vectorization for the combined section 

'''
#spilt data into train and test and validation sets
x_var = hyp_data.drop("Syllables", axis = 1)
y_var = hyp_data [['Syllables']]  
x_main, x_test, y_main, y_test = train_test_split(x_var, y_var, test_size = 0.2) 
x_train, x_val, y_train, y_val = train_test_split(x_main, y_main, test_size = 0.2)

#print ("the test data x size is: " + str(len(x_test)))
#print ("the test data y size is: " + str(len(y_test))) 

print(x_var.dtypes) 
print(y_var.dtypes) 
'''
#create a text vectorizatoin layer  
'''
vector_test = x_train["Original Raw"].iloc[1]
print(vector_test) 
#unique_chars = set(''.join(df['column_name'].astype(str))) for an entire column
unique_char_test = len(set(''.join(vector_test))) 
print (unique_char_test)
max_len_test = len(vector_test)
print (max_len_test)   

x_val.to_csv('helpx.csv')
y_val.to_csv('helpy.csv')
'''
'''
def vectorize(word, max_unique_chars, max_sequence_length):  
    vectorize_layer= TextVectorization( 
        max_tokens = max_unique_chars,
        output_mode='int', #change to binary if u wanna use one hot encoding instead
        output_sequence_length = max_sequence_length,
        split='character' #by char since we're doing words not sentences and stuff 
    ) 
    vectorize_layer.adapt(word) 
    return vectorize_layer(word).numpy() 

print(vectorize(vector_test, unique_char_test, max_len_test))

'''








'''
#tokenize the data and also count the vocab size and seuqence length 
#first, for the raw original 
unique_chars_raw = set(''.join(str(hyp_data['Original Raw']))) 
vocab_size_raw = len(unique_chars_raw)
max_word_length_raw = hyp_data['Original Raw'].str.len().max()  
hyp_data['Tokenized Original Raw'] = vectorize(hyp_data['Original Raw'], vocab_size_raw, max_word_length_raw)




def vectorize(word, vocab_size, sequence_length):  
    #vocab size equal to total unique characters in dataset 
    sequence_length = int(sequence_length)
    vectorize_layer= TextVectorization( 
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length,
        split='character' #by char since we're doing words not sentences and stuff 
    ) 
    vectorize_layer.adapt(word.batch(64)) 
    return vectorize_layer(word).numpy()
'''