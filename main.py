from flask import Flask, request, jsonify
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from PIL import Image
import requests

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello from Flask!'


@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url')
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Get the input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the new image
    img = Image.open(requests.get(url, stream=True).raw)
    # img = image.load_img("1163.jpg", target_size=(224, 224))
    x = image.img_to_array(img)
    resize = cv2.resize(x, (224, 224))
    x = np.expand_dims(resize, axis=0)
    x = preprocess_input(x)

    # Extract features from the new image
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    features = interpreter.get_tensor(output_details[0]['index'])
    print(features)

    dataset_features = np.array(pickle.load(open('embedding1.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))

    # Compute the cosine similarity between the new image's features and the features of all images in the dataset
    similarities = cosine_similarity(features, dataset_features)

    # Find the most similar images
    indices = similarities.argsort()[:, -6:][:, ::-1]

    links={}
    links["result"] = []
    for file in indices[0][1:5]:
        print(file)
        print(filenames[file])
        url = filenames[file]
        # links["result"].append(nums_from_string.get_nums(url)[0])
        links["result"].append(str(file))

    return jsonify(links)


if __name__ == '__main__':
    app.run(debug=True)
