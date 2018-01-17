# REF [site] >> http://pillow.readthedocs.io/en/3.1.x/reference/Image.html

from PIL import Image

img_filename = 'D:/dataset/pattern_recognition/street1.png'
#img_filename = 'D:/dataset/pattern_recognition/street2.jpg'

img = Image.open(img_filename)

img.rotate(45).show()

img.save('tmp.jpg')
