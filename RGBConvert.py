
import os
from PIL import Image

def is_I(image_path):
    img = Image.open(image_path)
    return img.mode == "I"

test_grayscale = "/home/zyvasheikh/Desktop/work/images HITL/spheroid HITL 4C/valid/70-100%"
test_RGB = "/home/zyvasheikh/Desktop/work/images HITL/spheroid HITL 4C/RGBvalid/70-100%"

os.makedirs(test_RGB, exist_ok=True)

for file_name in os.listdir(test_grayscale):
    file_path = os.path.join(test_grayscale,file_name)

    if os.path.isfile(file_path) and file_name.lower().endswith(('.png','.jpg','.jpeg')):

        if is_I(file_path):
            img = Image.open(file_path)
            
            img_8 = img.point(lambda i: i * (1/255))
            img_rgb = img_8.convert("RGB")
            
            output_file_path = os.path.join(test_RGB, file_name)
            img_rgb.save(output_file_path)

        else:
            output_file_path = os.path.join(test_RGB, file_name)
            os.replace(file_path, output_file_path)


print("conversion complete")

