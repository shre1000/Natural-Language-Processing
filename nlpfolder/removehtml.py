import os

'''
modify path as per current location. run this again while making new documents version. so download the part 1 then run this. have part 1 in current location.
'''

path = '/Users/User/Documents/Part1'

for root, dirs, files in os.walk(path):
    for file in files:
    	if file.endswith(".html"):
    		filepath = root + os.sep + file
    		os.remove(filepath)