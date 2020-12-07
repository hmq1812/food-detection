from yolo import DETECTMODEL
from PIL import Image
import os

model = DETECTMODEL()

input_dir = './input/'
output_dir = './output/'
for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(input_dir, filename)
        img = Image.open(path)   
        out_img = model.detect_image(img)
        out_img.save(output_dir + filename)

# image = Image.open('/home/hmquan/Downloads/test.jpg')
# result = model.detect_image(image)
# result.save('/home/hmquan/Downloads/rs.jpg')
