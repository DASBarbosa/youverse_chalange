from fastapi import FastAPI
from apis.inference_api import InferenceAPI
from loaders.image_loader import ImgLoaderTypes, create_img_loader
from loaders.model_loader import ModelLoader

app = FastAPI(title="Inference Service")
# The fact we built a factory for both image loader and model loader is pretty usefull here
# This means we can easily add more loaders without changing the api class signature
# The make this more dinamic in a real enviornment, we could inject the string containing
# the type of loader at deployment time as an env variable, then use that as imput for both factories
image_loader = create_img_loader(ImgLoaderTypes.OcvImageLoader)
model_loader = ModelLoader()

# Injectcting the app, model_loader and image loader in the InferenceAPI class, 
# allows us to instanciate both loaders at app initialization time
InferenceAPI(app = app, model_loader = model_loader, image_loader= image_loader)