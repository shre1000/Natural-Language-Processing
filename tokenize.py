from nltk.tokenize import sent_tokenize
import re


with open("C:\CSCI517\input.txt", "r") as myinputfile:
	data = myinputfile.read()
	
myinputfile.close()

sent_tokenize_list = sent_tokenize(data)

a = len(sent_tokenize_list)

print (a)

myoutputfile = open("C:\CSCI517\output1.txt", "w")

#i = 1

for s in sent_tokenize_list:
	myoutputfile.write(s+'\n')
	#myoutputfile.write((str(i))+"."+" "+s+'\n')
	#i = i + 1
	
myoutputfile.close()
	



