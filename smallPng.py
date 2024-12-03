from PIL import Image
import os
png_dir = "./rgb_png/"
res_dir = "./small_png"

for folder in os.listdir(png_dir):
    for png in os.listdir(png_dir + folder):
        path = os.path.join(png_dir, folder, png)
        image = Image.open(path)
        image = image.resize((100, 100))

        res_folder = os.path.join(res_dir, folder)
        if not os.path.exists(res_folder):
            os.mkdir(res_folder)
        res_path = os.path.join(res_folder, png)
        image.save(res_path)