from PIL import Image, ImageOps
import numpy as np

def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'potatoes')
    model.eval()
    return model

def teachable_machine_classification(img,potatoes):
    # Load the model
    model=load_model()

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (180,180)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability
