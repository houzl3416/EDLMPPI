
"""
conda install python==3.8.12
pip install tensorflow==2.4.1
pip install keras==2.4.3
pip install numpy==1.19.0
"""

import numpy as np
from keras.models import load_model
from utils import softmax
from capsule import Capsule


model = load_model('./model/0.h5',custom_objects = {"Capsule": Capsule})

vec = np.load('./example/ProtT5.npy')
MBF = np.load('./example/win_25.npy')

result = model.predict([vec.reshape(-1,1,1024),MBF])
print(result)

