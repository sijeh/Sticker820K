import os

import pkg_resources
from setuptools import setup, find_packages

packages = find_packages(exclude=["tests*"])
# with open('README_En.md', 'r', encoding='utf-8') as fp:
#     long_description = fp.read()
setup(name="sticker_clip",
      py_modules=["sticker_clip"],
      version="1.0",

      author_email="",
      long_description="",
      long_description_content_type="text/markdown",
      packages=packages,
      keywords='clip',
      install_requires=[
          str(r) for r in pkg_resources.parse_requirements(open(os.path.join(os.path.dirname(__file__), "requirements.txt")))
      ],
      data_files=[('clip/model_configs', [
          'sticker_clip/clip/model_configs/RoBERTa-wwm-ext-base-chinese.json',
          'sticker_clip/clip/model_configs/RoBERTa-wwm-ext-large-chinese.json', 'sticker_clip/clip/model_configs/ViT-B-16.json',
          'sticker_clip/clip/model_configs/ViT-B-32.json', 'sticker_clip/clip/model_configs/ViT-L-14.json',
          'sticker_clip/clip/model_configs/ViT-L-14-336.json', 'sticker_clip/clip/model_configs/ViT-H-14.json',
          'sticker_clip/clip/model_configs/RN50.json', 'sticker_clip/clip/model_configs/RBT3-chinese.json'
      ]), ('clip/', ['sticker_clip/clip/vocab.txt'])],
      include_package_data=True,
      description='Sticker CLIP.')
