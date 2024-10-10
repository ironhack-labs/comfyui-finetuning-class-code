#Ideally, the dataset should have square images. For that we can use this script.
from PIL import Image
import os

input_folder = "../assets/fine-tuning-imgs"
output_folder = "../assets/fine-tuning-imgs_resizes"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        img = Image.open(os.path.join(input_folder, filename))
        img = img.resize((512, 512))
        img.save(os.path.join(output_folder, filename))
