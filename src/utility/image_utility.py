# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2022-2023 Alexander Hering        *
****************************************************
"""
import os
from PIL import Image
import logging
from typing import Any, List, Optional
LOGGER = logging.Logger("[ImageUtility]")


IMAGE_EXTENSIONS = [".jpeg", ".jpg", ".png"]


def check_image_health(file_path: str) -> bool:
    """
    Function for checking image file health.
    Taken from @https://github.com/ftarlao/check-media-integrity and adjusted.
    :param file_path: File path of image file to check.
    :return: True, if image file is healthy, else False.
    """
    try:
        if not os.path.exists(file_path):
            LOGGER.warning(f"Could not find '{file_path}'!")
            return False
        elif os.path.splitext(file_path)[1].lower() not in IMAGE_EXTENSIONS:
            LOGGER.warning(f"File extension is not in {IMAGE_EXTENSIONS}!")
            return False
        img = Image.open(file_path)
        img.verify()
        img.close()
        img = Image.open(file_path) 
        img.transpose(Image.FLIP_LEFT_RIGHT)
        img.close()
        LOGGER.info(f"'{file_path}' is valid.")
        return True
    except: 
        LOGGER.warning(f"'{file_path}' is corrupted!")
        return False
