# CSCI 517: Assignment 2:  You Write I Fix


import re
import nltk
import collections
from nltk.tokenize import word_tokenize
from collections import Counter
import sys
import os

# Reading dataset.
input_data = open("dataset.txt", "r")
data = input_data.read()
# Converting all words to lower case.
data = data.lower()
input_data.close()

# Tokenize dataset.
tokens = nltk.word_tokenize(data)

# Creating unigram model.
unigram = Counter(tokens)

# Asking for file path from user.
user_input = input("Enter the path of your file: ")
assert os.path.exists(user_input), "File not found at, "+str(user_input)

# Reading input file.
input_file = open(user_input, "r")
text = input_file.read()
input_file.close()

# Tokenize input file.
text = nltk.word_tokenize(text)

# Creating symbol set so that they don't need to pass through spell corrector.
symbol_set = set(['!', '@', '#', '$', '%', '^', '&', '*', '+', ':', ';', ',', '.', '?', '/'])

# Creating output list to write in output file.
output_list = []

# Returns the subset of words that appeared in the dataset i.e. unigram dictionary.
def known_words(words): 
    return set(w for w in words if w in unigram)

# Returns all edits that are one word away from the word.
def edits_with_cost_1(single_word):
    letters    		= 'abcdefghijklmnopqrstuvwxyz'
    split      		= [(single_word[:i], single_word[i:])    for i in range(len(single_word) + 1)]
    delete     		= [L + R[1:]               				 for L, R in split if R]
    reverse    		= [L + R[1] + R[0] + R[2:] 				 for L, R in split if len(R)>1]
    substitution   	= [L + c + R[1:]           				 for L, R in split if R for c in letters]
    insert     		= [L + c + R               				 for L, R in split for c in letters]
    return set(delete + reverse + substitution + insert)

# Returns all edits that are two edits away from word.
def edits_with_cost_2(single_word): 
    return (e2 for e1 in edits_with_cost_1(single_word) for e2 in edits_with_cost_1(e1))

# Generate possible candidate words for given word.
def candidate_words(single_word): 
    return (known_words([single_word]) or known_words(edits_with_cost_1(single_word)) or known_words(edits_with_cost_2(single_word)) or [single_word])
	
# Calculating probability of a word.
def probability(single_word, N=sum(unigram.values())): 
    return unigram[single_word] / N

# Gives most correct word for given word.
def correct_word(single_word): 
    return max(candidate_words(single_word), key=probability)
	
# Correcting each word.
def correction_of_text(text):
	# Go through each word and correct each word.
	for i in range(0, (len(text))):
		if(text[i] in symbol_set):
			# If it is a symbol then return as it is and continue.
			output_list.append(text[i])
			continue	
		else:
			# Convert word to lower case.
			single_word = text[i].lower()
			corrected_spelling = correct_word(single_word)
			if (corrected_spelling == single_word):
				# If corrected spelling and original word is same then just add word to output_list.
				output_list.append(" "+text[i])
			else:
				# If corrected spelling is different than original then add original word along with corrected spelling in brackets to output_list.
				output_list.append(" "+text[i]+"("+corrected_spelling+")")


# Calling correction method to correct input file.
correction_of_text(text)

# Open output file and write output_list to it.
with open("output.txt", "w") as f:
	f.writelines(output_list)
			
		
		

	

