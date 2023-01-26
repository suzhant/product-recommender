import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from tensorflow.keras.applications import vgg16


# Transfer Learning for feature extraction (ResNet50) and KNN for prediction/recommendation
#model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable=False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# print(model.summary())
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

filenames = []

for i, file in enumerate(os.listdir('images')):
    if i >= 2000:
        break
    filenames.append(os.path.join('images', file))


feature_list = []

for file in tqdm(filenames):
    feature=extract_features(file, model)
    feature_list.append(feature)

pickle.dump(feature_list, open('embedding5.pkl', 'wb'))
pickle.dump(filenames, open('filename5.pkl', 'wb'))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('model5.tflite', 'wb') as f:
    f.write(tflite_model)

