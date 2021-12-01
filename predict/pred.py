
"""
conda install python==3.8.12
pip install tensorflow==2.4.1
pip install keras==2.4.3
pip install numpy==1.19.0
"""

import numpy as np
from keras.models import load_model
from predict.utils import softmax
from predict.capsule import Capsule


model = load_model(base_url + 'predict/model/1.h5',custom_objects = {"Capsule": Capsule})

vec = np.load(base_url)

result = model1.predict(vec.reshape(-1,1,1024))


