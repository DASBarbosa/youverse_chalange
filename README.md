# Youverse chalange

## Installing dependencies
To install all need python dependencies, make sure to have python3.12 installed. Use the provided [requirements.txt](requirements.txt) to install all the relevent python packages.

`
pip install -r requirements.txt
`

Note: Idealy, you should do this in a virtual enviornment so it does not conflicts with your global pip installations. 

The packages that were manualy installed with pip were the following, the rest are dependencies of these packages:

- numpy: numpy is the base for data that is infered by the models
- onncruntime: the runtime requested to be used by this exercice
- pydantinc: pydentic is a personal choice for creating data models, it's json serialization and deserialization capabilities are specially usefull when building rest api endpoints
- black: installed just to help with formating
- cv2: used to load the images locally for the initial tests with the model
- fastapi: The selected framework to build a puthon api