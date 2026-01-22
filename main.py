from fastapi import FastAPI
from apis.inference_api import InferenceAPI

app = FastAPI(title="Inference Service")

# Inject app into API class
InferenceAPI(app)