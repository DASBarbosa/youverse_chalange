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

 This api does not require too much explanation, it has a event handler on start up that starts the image loader and the model loader, it also loads the model. Depending on the sucess of this initialization, it sets some health metrics that will be available on the /health endpoint.

 The /infer endpoint, receives an image in bytes, uses the image loader to conver it to a numpy array and passes it to model loader to infer the image. This is the only endpoint that uses async because it handles I/O tasks, mainly loading image bytes over a network call. async lets Python "pause" a function while it’s waiting on something and do other work instead. In this particular case what happens is:

 - A request comes in
 - FastAPI starts handling it
 - It hits await image.read()
 - While the image is being read from the client FastAPI pauses this request
 - The event loop serves other requests
 - When the read finishes, execution resumes

 without async, one slow client uploading a big image could block api calls or cause latancy spikes.

 On a kubernetes deployment fase this can be further improved with what we call replicas of the same micro service, improving redundancy and load balancing.

 There are two key choices in terms if libraries that I would like to highlight here.
 
 The use of pydantic BaseModels helps strongly type the data that's flowing in and out of the API, it works realy well with FastAPI, since BaseModels have built in json serialization and deserialization methods.

 The use of pydantic BaseSettings helps to strongly type the settings data that are being injected on the API, on top of that, BaseSettings can automatically convert env variables into the class parameters.

 ### Utils

 This module was created to store some scripts that are generic enough that can be reused across other python modules. In this small project I just created 2 examples:

 - A [stdin_loader.py](utils/stdin_loader.py) that has utils to deal with console input data, this was not really needed for the exercice, but it helped me in a first step while I was testing the onnx model without having an api built.
 - A [time_utils.py](utils/time_utils.py) that has a context manager to measure the time a task takes to execute. This was something requested on this exercice for mesuring the time the inference took. I thought about transforming this time measurement into a context manager because it would be easier to reuse in multiple places without the need to allways create new dedicated variables to measure time and always manually calculate the time enlapsed. Initially I was also thinking in using this to measure time with and without threads, but I ended up not having time to implement threads and run this expirement.

## Future work
Troughout this document there were already mentioned many ways on how this implementation could be better, but here is a breif summary of what already has been stated and maybe some more considerations:

- Usages of multythreadng in I/O tasks, or tasks that make good use of the Global Interpreter Lock (GIL), such as the image processing tasks that use numpy, or the inference tasks.
- Allow a different model to be loaded without restarting the service. In the current implementation the model is loaded when a instance of the model_loader class is instanciated. This turned out to be more of a architectural
decision than I anticipated, meaning that I would have to change things in various places if the implementation to accomodate for this functionality. While these changes are not particular hard to do, I noticed this late in the implementation, which led me to not add this functionality for the sake of time.
- Speaking of loading the model after the service has started, if this was the intent, maybe the use of env variables to specify the path to the module would not be the best approach, specially in a cloud enviornment were it would either require a redeployment to update those envs, or actually entering inside the service container and manualy do it. Maybe having an enum with all the possible models and allowing the user to choose which one to use on the request would be a possibility? or a dedicated service for serving the available models in the backend, with an interface for the developers to switch the current active model in case we don't really want to allow users of the api to decide that? This point is a bit tricky as it would requireme to actually learn more abot the companny usecases and how services are handled.
- I'm not too familiar with the usage of this type of tools for image inference, but from what I have read, batch inference can help speed up the process when running these types of services using a GPU. If handled properly it would be much more efficient for a user to send multiple images rather then one by one.
- Usage of MakeFiles for local setup. This is a personal preference, but on real projects, everything that requires setup to run locally, I like to create make targets to implement them. For example, in this scenario having a target were you could just call `make setup` and it would automatically create a venv for you and install all the python dependencies would be pretty helpfull.  