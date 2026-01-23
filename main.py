from fastapi import FastAPI
from apis.inference_api import InferenceAPI
from loaders.image_loader import ImgLoaderTypes, create_img_loader
from loaders.model_loader import create_model_loader, ModelLoaderTypes

app = FastAPI(title="Inference Service")
# The fact we built a factory for both image loader and model loader is pretty usefull here
# This means we can easily add more loaders without changing the api class signature
# The make this more dinamic in a real enviornment, we could inject the string containing
# the type of loader at deployment time as an env variable, then use that as imput for both factories
img_loader_type = ImgLoaderTypes.OcvImageLoader
model_loader_type = ModelLoaderTypes.OnnxLoader
InferenceAPI(app = app, img_loader_type = img_loader_type, model_loader_type= model_loader_type)