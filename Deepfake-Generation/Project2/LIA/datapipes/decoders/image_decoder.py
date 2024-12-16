##############################################################################
# Copyright (c) 2022 DFKI GmbH - All Rights Reserved
# Written Stephan Krauß <Stephan.Krauss@dfki.de>, February 2023
# and by Suresh Guttikonda <Suresh.Guttikonda@dfki.de>, February 2023
# based on https://github.com/pytorch/pytorch/blob/master/torch/utils/data/datapipes/utils/decoder.py#L114
##############################################################################
"""doc
# datapipes.utils.image_decoder

> Decode images to numpy arrays, torch tensors or PIL images.

"""
from __future__ import annotations
import io
from imageio import mimread
import numpy as np
import torch
#from ldm.modules.face_mesh3_efficient_3D_conv import MeshingFace
import cv2

image_specs = {
    "u8": ("numpy", "uint8", "raw"),
    "l8": ("numpy", "uint8", "l"),
    "rgb8": ("numpy", "uint8", "rgb"),
    "rgba8": ("numpy", "uint8", "rgba"),

    "u": ("numpy", "float", "raw"),
    "l": ("numpy", "float", "l"),
    "i": ("numpy", "float", "i"),
    "rgb": ("numpy", "float", "rgb"),
    "rgba": ("numpy", "float", "rgba"),

    "torchu8": ("torch", "uint8", "raw"),
    "torchl8": ("torch", "uint8", "l"),
    "torchrgb8": ("torch", "uint8", "rgb"),
    "torchrgba8": ("torch", "uint8", "rgba"),

    "torchu": ("torch", "float", "raw"),
    "torchl": ("torch", "float", "l"),
    "torchi": ("torch", "float", "i"),
    "torch": ("torch", "float", "rgb"),
    "torchrgb": ("torch", "float", "rgb"),
    "torchrgba": ("torch", "float", "rgba"),

    "pilu": ("pil", None, "raw"),
    "pill": ("pil", None, "l"),
    "pili": ("pil", None, "i"),
    "pil": ("pil", None, "rgb"),
    "pilrgb": ("pil", None, "rgb"),
    "pilrgba": ("pil", None, "rgba"),
}


class ImageDecoder:
    """
    A custom variant of pyTorch's ImageHandler.

    It adds support for 16 bit single channel images and the WebP image format. Furthermore, it ensures that single
    channel images always have a channel dimension when converting them to numpy or torch. This fixes a crash when
    trying to load single channel images as torch tensors.

    How the image data shall be decoded is defined by the given `image_spec`.
    The `image_spec` specifies whether
    (1) the image is returned as a numpy array, torch tensor or PIL image,
    (2) the data type of the elements in the numpy array or torch tensor is uint8 or float32,
    (3) the image is converted to 8-bit grayscale (l), 16-bit grayscale (i), 8-bit RGB (rgb), 8-bit RGBA (rgba), or
        kept in the same format as in the image file (raw).

    The available choices are:
    - u8: numpy uint8 raw
    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - u: numpy float raw
    - l: numpy float l
    - i: numpy float i
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - torchu8: torch uint8 raw
    - torchl8: torch uint8 l
    - torchrgb8: torch uint8 rgb
    - torchrgba8: torch uint8 rgba
    - torchu: torch float raw
    - torchl: torch float l
    - torchi: torch float i
    - torch: torch float rgb
    - torchrgb: torch float rgb
    - torchrgba: torch float rgba
    - pilu: pil None u
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba
    """
    def __init__(self, image_spec: str):
        # assert image_spec in list(image_specs.keys()), "unknown image specification: {}".format(image_spec)
        self.image_spec = image_spec.lower()

    def __call__(self, extension: str, data: bytes):
        if extension.lower() not in "jpg mp4 jpeg png ppm pgm pbm pnm tga tiff webp".split():
            return None

        try:
            import PIL.Image
        except ImportError as e:
            raise ModuleNotFoundError("Package 'PIL' is required to be installed to decode images."
                                      "Please use `pip install Pillow` to install the package") from e


        with io.BytesIO(data) as stream:
            video = np.array(mimread(stream,memtest=False, format='mp4'))

            return video
            # image = PIL.Image.open(stream)
            # image.load()

            # if mode != "raw":
            #     image = image.convert(mode.upper())

            # if a_type == "pil":
            #     return image

            # image = np.asarray(image, dtype=np.uint16 if image.mode == "I" else np.uint8)
            # # add missing channel dimension if case of single channel (grayscale) images

class VideoDecoder:

    def __init__(self):
        pass
        #self.MeshingFace = MeshingFace()

    def __call__(self, extension: str, data: bytes):

        with io.BytesIO(data) as stream:
            video = np.array(mimread(stream,memtest=False, format='mp4'))
            frame_num = np.sort(np.random.choice(video.shape[0], replace=True, size=2))
            drive_frame = video[frame_num[1]]
            source_frame = video[frame_num[0]]

            return video[frame_num]
