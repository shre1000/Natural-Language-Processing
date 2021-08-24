# Importing regular expression package.
import re

# Declaring data variable.
data = ""

# Opening file and reading it.
with open("input.txt", "r") as myinputfile:
	for line in myinputfile.readlines():
		if re.search('\S', line): 
			data = data + line
		
# Closing file.
myinputfile.close()

# Regular expression to perform sentense segmentation.
sentenses = re.split('(?<!\w\.\w\.)(?<!\w\.\w\.\w\.\w\.)(?<![A-Z]\.)(?<!MR.|Mr.|\.\.\.)(?<!\;\"|\,\")(?<=\.|\?|\]|\")\s' , data)

# Opening output file.
myoutputfile = open("output.txt", "w")

# Declairing counter for sentense numbers.
i = 1

# Writing to output file.
for s in sentenses:
	myoutputfile.write((str(i))+"."+" "+s+'\n')
	i = i + 1

# Closing output file.
myoutputfile.close()
