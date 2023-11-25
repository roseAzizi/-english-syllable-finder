import pandas as pd  
import tensorflow as tf 
#from tensorflow.python.keras.preprocessing.sequence import pad_sequences  
#from tensorflow.python.keras.layers import TextVectorization

def main(): 
    #change float to int at the end of the script if ur running into memory issues
    column_specs = { 
        'Original Raw': str,  
        'Tokenized Original Raw': object,
        'Regular Word': str,  
        'Tokenized Regular Word': object,
        'Syllables': float, 
        'Word Length': float, 
        'Vowel Constonant Pattern': str,  
        'Tokenized Vowel Constonant Pattern': object,
        "Total Number of Vowels": float
    }
    hyp_data = pd.read_csv('mhyph.txt', sep=r'\s{2,}', encoding='iso-8859-15', engine = 'python', header = None, 
    names=list(column_specs.keys()), dtype=column_specs)  

    hyp_data.drop_duplicates()  

    feature_extraction(hyp_data)

    #pickle it for later
    hyp_data.to_pickle('hyp_data.pkl') 


def feature_extraction(hyp_data):
    #im going to replace spaces with "-" to make things easier in the tokenization step  
    #idk if this is an issue in modifying the original raw data 
    #keeping these comments in case i need to make a new column with these changes instead later
    hyp_data['Original Raw'] = hyp_data['Original Raw'].str.replace(" ", "-", regex = True)  
    #syllables df counts the seperators, spaces and hypens 
    hyp_data['Syllables'] = hyp_data['Original Raw'].str.count(r'[¥ -]') + 1  
    #remove the seperators 
    hyp_data['Regular Word'] = hyp_data['Original Raw'].str.replace('¥','',regex=False) 
    #word length 
    hyp_data['Word Length'] = hyp_data['Regular Word'].str.len() 
    #VC pattern 
    hyp_data['Vowel Constonant Pattern'] = hyp_data['Original Raw'].apply(vowel_consonant_pattern)
    # Count total number of vowels in the pattern
    hyp_data['Total Number of Vowels'] = hyp_data['Vowel Constonant Pattern'].str.count('V')
    #helps with memory and is easy 
    #hyp_data['Syllables'] = hyp_data['Syllables'].astype(int)
    #hyp_data['Word Length'] = hyp_data['Word Length'].astype(int)
    #hyp_data['Total Number of Vowels'] = hyp_data['Total Number of Vowels'].astype(int)


#function to find the vowel constoant patter
def vowel_consonant_pattern(word):
    word = str(word).lower() 
    pattern = ''
    for j, i in enumerate(word):
        if i in "aeiou":
            pattern += "V" 
        #and sometimes y too :))
        elif i == "y": 
            #start of words and syllables
            if j == 0 or (j>0 and word[j-1] in '¥ -'):
                pattern += "C"
            else:
                pattern += "V"
        elif i in '¥ -':
            pattern += " "
        elif i.isalpha():
            pattern += "C"
    return pattern

#def pad_and_token(): 
    #tokenizer = Tokenizer(char_level=True)  


if __name__ == "__main__":
    main()


