from fastai import *
from fastai.vision import *

#help(untar_data)

'''
Help on function untar_data in module fastai.datasets:

untar_data(url:str, fname:Union[pathlib.Path, str]=None, dest:Union[pathlib.Path, str]=None)
Download `url` if doesn't exist to `fname` and un-tgz to folder `dest`
'''
data_url = URLs.PETS

path = untar_data(data_url)

print(path.ls())

path_anno = path/'annotations'
path_img = path/'images'
