import os

'''
Change the path directory depending on folder. modify abstract with caps, lower case, DCescription terms. modify not available to caps etc. and '' to ' '.  run this file again.
'''
path = '/Users/User/Documents/Part1'

data = " "

for root, dirs, files in os.walk(path):
    for file in files:

    	filepath = root + os.sep + file
    	filename = filepath
 
    	f = open(filename, "r")
    	
        for line in f:
        	if 'Abstract' in line:
				for line2 in f: 
					line2 = line2.rstrip()
					line2 = line2.strip() + " "
					data = data + line2
				data = data + '\n'
				break
		
modified_data = data.split("\n")

modified_data. remove('Not Available ')

modified_data. remove('')

with open('documents.txt', 'w') as f:
    for item in modified_data:
        f.write("%s\n" % item)


