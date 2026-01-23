from fastapi import FastAPI
from apis.inference_api import InferenceAPI, InferenceApiSettings
from loaders.image_loader import ImgLoaderTypes, create_img_loader
from loaders.model_loader import create_model_loader, ModelLoaderTypes

app = FastAPI(title="Inference Service")
# The fact we built a factory for both image loader and model loader is pretty usefull here
# This means we can easily add more loaders without changing the api class signature
# What makes this more dinamic in a real enviornment, is we can  inject the string containing
# the type of loader at deployment time as an env variable
api_Settings = InferenceApiSettings() #When initialized with no arguments BaseSettings will look for env variables
InferenceAPI(app = app, api_settings=api_Settings)