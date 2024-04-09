from urllib.request import Request

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import FileResponse
import io
import pickle
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

animal_model = load_model("models/animal.h5", compile=True)
animal_input_shape = animal_model.layers[0].input_shape[0]
animal_classes = {0: 'antelope', 1: 'badger', 2: 'bat', 3: 'bear', 4: 'bee', 5: 'beetle', 6: 'bison', 7: 'boar',
                  8: 'butterfly', 9: 'cat', 10: 'caterpillar', 11: 'chimpanzee', 12: 'cockroach', 13: 'cow',
                  14: 'coyote',
                  15: 'crab', 16: 'crow', 17: 'deer', 18: 'dog', 19: 'dolphin', 20: 'donkey', 21: 'dragonfly',
                  22: 'duck',
                  23: 'eagle', 24: 'elephant', 25: 'flamingo', 26: 'fly', 27: 'fox', 28: 'goat', 29: 'goldfish',
                  30: 'goose',
                  31: 'gorilla', 32: 'grasshopper', 33: 'hamster', 34: 'hare', 35: 'hedgehog', 36: 'hippopotamus',
                  37: 'hornbill', 38: 'horse', 39: 'hummingbird', 40: 'hyena', 41: 'jellyfish', 42: 'kangaroo',
                  43: 'koala',
                  44: 'ladybugs', 45: 'leopard', 46: 'lion', 47: 'lizard', 48: 'lobster', 49: 'mosquito', 50: 'moth',
                  51: 'mouse', 52: 'octopus', 53: 'okapi', 54: 'orangutan', 55: 'otter', 56: 'owl', 57: 'ox',
                  58: 'oyster',
                  59: 'panda', 60: 'parrot', 61: 'pelecaniformes', 62: 'penguin', 63: 'pig', 64: 'pigeon',
                  65: 'porcupine',
                  66: 'possum', 67: 'raccoon', 68: 'rat', 69: 'reindeer', 70: 'rhinoceros', 71: 'sandpiper',
                  72: 'seahorse',
                  73: 'seal', 74: 'shark', 75: 'sheep', 76: 'snake', 77: 'sparrow', 78: 'squid', 79: 'squirrel',
                  80: 'starfish', 81: 'swan', 82: 'tiger', 83: 'turkey', 84: 'turtle', 85: 'whale', 86: 'wolf',
                  87: 'wombat',
                  88: 'woodpecker', 89: 'zebra'}

plant_model = load_model("models/plant.h5", compile=True)
plant_input_shape = plant_model.layers[0].input_shape
print(plant_input_shape)
with open('models/plant_label_transform.pkl', 'rb') as f:
    plant_label_transform = pickle.load(f)


# Define the /prediction route
@app.post('/predict/animal')
async def predict_animal(file: UploadFile = File(...)):
    # Ensure that this is an image
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    # Read image contents
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))

    # Resize image to expected input shape
    pil_image = pil_image.resize((animal_input_shape[1], animal_input_shape[2]))

    # Convert image into numpy format
    numpy_image = np.array(pil_image).reshape((animal_input_shape[1], animal_input_shape[2], animal_input_shape[3]))

    # Generate prediction
    prediction_array = np.array([numpy_image])
    predictions = animal_model.predict(prediction_array)
    prediction = predictions[0]
    likely_class = np.argmax(prediction)

    return {"result": animal_classes[likely_class]}


@app.post('/predict/plant')
async def predict_plant(file: UploadFile = File(...)):
    # Ensure that this is an image
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    # Read image contents
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))

    # Resize image to expected input shape
    pil_image = pil_image.resize((plant_input_shape[1], plant_input_shape[2]))
    numpy_image = np.array(pil_image).reshape((plant_input_shape[1], plant_input_shape[2], plant_input_shape[3]))
    numpy_image = numpy_image / 255
    numpy_image = np.expand_dims(numpy_image, axis=0)

    predicted_probabilities = plant_model.predict(numpy_image)
    predicted_labels = plant_label_transform.inverse_transform(predicted_probabilities)

    return {"result": predicted_labels[0]}


@app.get("/")
async def index():
    return FileResponse("index.html")
