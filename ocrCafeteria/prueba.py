from PIL import Image
from pytesseract import *
 
#image_file = 'menu.jpg'
#im = Image.open(image_file)
#text = image_to_string(im)
#text = image_file_to_string(image_file)
#text = image_file_to_string(image_file, graceful_errors=True)

#print(pytesseract.image_to_string(Image.open('menu.jpg')))
#print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))
print '#################### MENU #################'
print(pytesseract.image_to_string(Image.open('threshMagro.jpg'), lang='spa'))
print '-----------------------------------------'
#print(pytesseract.image_to_string(Image.open('letras.png'), lang='spa'))

#print "=====output=======\n"
#print text
