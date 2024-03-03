from utils import *
import numpy as np

import torch
from torchvision import transforms
from PIL import Image

def stegano_solver(cover_im: np.ndarray, message: str) -> str:
    # image_path = "./sample_example/encoded.png"
    # image = Image.open(image_path)
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    # ])
    #
    # image_tensor = transform(image)
    # print(image_tensor.shape)
    # make_message(image_tensor)

    # Assuming cover_im is a 3D numpy array representing the image
    # Convert it to a PIL Image

    image_path = "./sample_example/encoded.png"
    cover_image = Image.open(image_path)

    # cover_image = Image.fromarray(cover_im.astype('uint8'), 'RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    # Apply the transformation
    image_tensor = transform(cover_image)

    # Add a batch dimension (unsqueeze) to match the expected shape
    image_tensor = image_tensor.unsqueeze(0)

    print(image_tensor.shape)

    # Now call the function with the corrected tensor
    return decode(image_tensor)



print(stegano_solver(np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]), "hello"))