import os

#count length of images

path = os.path.join(os.getcwd(),"images")

count = 0
# Iterate directory
for i in os.listdir(path):
    # check if current path is a file
    if os.path.isfile(os.path.join(path, i)):
        count += 1
print('File count:', count)


#for i in os.listdir(path):
    