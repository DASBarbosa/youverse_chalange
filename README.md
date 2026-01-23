# Youverse chalange

## Installing dependencies
To install all need python dependencies, make sure to have python3.12 installed. Use the provided [requirements.txt](requirements.txt) to install all the relevent python packages. Before installing the requirements it is recommended that you create a virtual enviornment. assuming the alias you have on your local machine for python3.12 is "python3.12" use the following command to create a venv:

`
python3.12 -m venv myenv
`

to activate the venv you just crated use:

`
source myenv/bin/activate
`

now install the python requirements with the following command, this assumes you are in the root folder of this repo:


`
pip install -r requirements.txt
`

Note: This could have been improved by using a tool like poetry or conda, but for this exercice I decided to keep it simple.

The packages that were manualy installed with pip were the following, the rest of them you see on the requirements file are dependencies of these packages:

- numpy: numpy is the base for data that is infered by the models
- onncruntime: the runtime requested to be used by this exercice
- pydantinc: pydentic is a personal choice for creating data models, it's json serialization and deserialization capabilities are specially usefull when building rest api endpoints
- pydantic-settings: for usinf BaseSetting to load env variables
- black: installed just to help with formating
- cv2: used to load the images locally for the initial tests with the model
- fastapi: The selected framework to build a puthon api
- python-multipart: required to run the api
- uvicorn: required to run the api
- pytest: for running tests
- httpx: required for TestClient to work

## Running the app

If you wish to run a quick test on the inference model, you can run the [run_locally.py](run_locally.py) script by providing the path to the image as an argument:

`
python run_localy.py "/absolute/path/to/img.png"
`

To run the full api localy on your terminal you first have to set up the needed enviormental variables, an [api_settings.env](api_settings.env) is already provided to make it easier for you to load these envs on your terminal:

`
export $(grep -v '^#' api_settings.env | xargs)
`

This is what the above command does:
 - grep -v '^#' .env -> ignores comment lines in your .env
 - xargs -> converts KEY=VALUE lines into arguments for export.
 - export $(...) -> sets all those variables in your current shell.

On the same console you ran the above command, run the following to start the api:

`
uvicorn main:app --reload
`

test the infer endpoint or your terminal using:

` curl -X 'POST' 'http://127.0.0.1:8000/infer' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'image=@/absolute/path/to/img.png;type=image/png'
`

or using swagger at: http://127.0.0.1:8000/docs

## Unit tests

For the tests, due to the lack of time, only 2 were provided:

- A test on the normalization of the image process, present in the [test_img_loader.py](tests/test_img_loader.py) file
- A test to the /health endpoint of the fast api, by mocking the fast api itself, using the InferenceAPI interface and using the TestClient class from fastapi.testclient to use this mock to test the api without having to actuall spin a server. you can look at these tests at [test_inference_api.py](tests/test_inference_api.py)

The main reason why no test were made for the model loader, was because I don't have much experience in using these types tools (tools relates to image classification) therefore I'm also not to expireinced in mocking such models. Due to the time constrains I opted to make a test for what I was more confortable with.

## Structure of the project

The project is composed by 3 main modules:
- A module for loaders
- A module for api specifications
- A module for utils

### Loaders

The loaders module, contains both the [image_loader.py](loaders/image_loader.py) and the [model_loader.py](loaders/model_loader.py) libraries

For both these loaders, I opted to use a classic design pattern called the factory pattern, because it makes it easier to add new types of loaders. With a factory, we can easily change the library or meodology we use for loading an image or for infering said image with a different model, without having to change anything in the api it self and maintaining the api signature.

On the image loader I opted to use opencv as it was the library I was most confortable with, as I have used it in the past, but like I mentioned above, with the factory patern it's really easy to add a different class that uses a diferent library to load images.

On the model loader, I used onnxruntime as requested. Altough I have not used multythreadingin this exercice, I do acknowledge that it could have helped both on the image processing and on the inference, I have not done any real test to prove this, but In CPython, only one thread executes Python bytecode at a time this is what is called the Global Interpreter Lock (GIL). This means even if you have multiple threads, only one thread can run Python code at any given moment. What this implies is:

- CPU-bound Python code (e.g., heavy number crunching in pure Python) does not get a speed-up from multithreading because threads are taking turns due to the GIL.

- I/O-bound tasks (file reading, network requests) can benefit from threads because while one thread waits for I/O, another can run.

 Apperently some libraries, like opencv, numpy and tensorflow for example, do release the GIL temporarly when they are doing computacional expensive tasks. In theory running multythreading on the image normalization or on the inference run, could have helped to speed up the process, because those libraries would release the GIL letting other tasks to run.

 ### APIS
 I opted to create a wrapper arround the app for FastApi, I think this is cleaner and it helps keep track of dependencies, and above all, makes it easier to test the api itself, as we can easily mock and inject data on the wrapper. 

 This api does not require too much explanation, it has a even handler on start up that starts the image loader and the model loader, it also loads the model. Depending on the sucess of this initialization, it sets some health metrics that will be available on the /health endpoint.

 The /infer endpoint, receives an image in bytes, uses the image loader to conver it to a numpy array and passes it to model loader to infer the image. 

 There are two key choices that I would like to highlight here 
