import numpy as np 
import pandas as pd 
import matplotlib as plt 

hyp_data = pd.read_csv('mhyph.txt', sep=r'\s{2,}', encoding='iso-8859-15', engine = 'python', header = None, names=['Original Raw', 'Regular Word', 'Syllables'])  
hyp_data = hyp_data.astype({"Original Raw": str })

for index, row in hyp_data.iterrows():   
    syllable_count = row['Original Raw'].count('¥') + row['Original Raw'].count(' ') + row['Original Raw'].count('-')+ 1  
    hyp_data.at[index, "Syllables"] = syllable_count 
    raw_word = row['Original Raw'].replace('¥','') 
    hyp_data.at[index, "Regular Word"] = raw_word

print ('test')


