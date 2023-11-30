import pandas as pd  


def main(): 
    #change float to int at the end of the script if ur running into memory issues
    column_specs = { 
        'Original Raw': str,  
        'Regular Word': str,  
        'Syllables': float, 
        'Word Length': float, 
        'Vowel Constonant Pattern': str,  
        'Total Number of Vowels': float
    }
    hyp_data = pd.read_csv('mhyph.txt', sep=r'\s{2,}', encoding='iso-8859-15', engine = 'python', header = None, 
    names=list(column_specs.keys()), dtype=column_specs)  

    hyp_data.drop_duplicates()  

    hyp_data = feature_extraction(hyp_data)

    #pickle it for later
    hyp_data.to_pickle('hyp_data.pkl') 
    print ("pickled and ready to use! :))")

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
    #remove any weird or non-words in the dataset. Basically anything without vowels or haivng /'s 
    hyp_data = hyp_data[~hyp_data['Regular Word'].astype(str).str.contains(r"[^a-zA-Z ']")]  
    hyp_data = hyp_data[hyp_data['Total Number of Vowels'] != 0]
    # Fill NaN values with 0 for selected columns
    hyp_data[['Syllables', 'Word Length', 'Total Number of Vowels']] = hyp_data[['Syllables', 'Word Length', 'Total Number of Vowels']].fillna(0)
    # Filter out rows where any of the specified columns are 0
    hyp_data = hyp_data[(hyp_data[['Syllables', 'Word Length', 'Total Number of Vowels']] != 0).all(axis=1)] 
    #make into int to make my life easier 
    hyp_data[['Syllables', 'Word Length', 'Total Number of Vowels']] = hyp_data[['Syllables', 'Word Length', 'Total Number of Vowels']].astype(int)
    #Combinde all the object for preprocessing 
    #hyp_data["Combinded Strings"] = hyp_data['Original Raw'] + ' ' + hyp_data['Regular Word'] + ' ' + hyp_data['Vowel Constonant Pattern']
    return hyp_data
   

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




if __name__ == "__main__":
    main()


