import imageio
import os

from src.config import GPT3_175B, LLAMA_1_7B, OPT_125M, W16A32, W4A16, W4A8, W8A8

# Define the directory containing the images and the output GIF filename
model = LLAMA_1_7B
quant = W8A8
image_list = [
    f"outputs/full_decode/{model.name}_{quant.name}_decode={idx}/interesting_layers_full.png"
    for idx in range(1025, 2048)
]

output_gif = "outputs/decoding.gif"


images = [imageio.imread(file) for file in image_list]
gif_duration = 10  # in seconds
time_per_frame = gif_duration / len(image_list)
imageio.mimsave(output_gif, images, duration=time_per_frame)  # duration is the time between frames in seconds
