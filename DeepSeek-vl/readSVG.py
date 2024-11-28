
import cairosvg
from PIL import Image
import numpy as np
import cv2
import os

def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype='float32') / 255.0
    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B


    return np.asarray(rgb, dtype='uint8')
# 读取SVG文件并转换为PNG

def svg2rgb(folder, file):
    png_dir = "../rgb_png"
    svg_dir = "../svg/vector"
    png_folder = os.path.join(png_dir, folder)
    png_file = file.split('.')[0] + '.png'
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)
    svg_path = os.path.join(svg_dir, folder, file)
    png_path = os.path.join(png_folder, png_file)
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    raw_image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    image = rgba2rgb(raw_image)
    cv2.imwrite(png_path, image)

# if __name__ == "__main__":
#     folder_dir = "./3D"
#     for file in os.listdir(folder_dir):
#         result = os.path.join("./rgb_png", file.split('.')[0] + '.png')
#         svg_file = os.path.join(folder_dir, file)
#         cairosvg.svg2png(url=svg_file, write_to=result)
#         raw_image = cv2.imread(result, cv2.IMREAD_UNCHANGED)
#         # print(raw_image.shape)
#         image = rgba2rgb(raw_image)
#         cv2.imwrite(result, image)