from capsule import Capsule
import keras
from keras.backend import sigmoid
from keras.layers import (LSTM, GRU, Dense, Dropout,concatenate,
                          Flatten, Input)
                          
from keras.layers import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model


def createmodel3(kmer):

    input1 = Input(shape=(1,1024), dtype="float", name='input1')

    xl_0 = BatchNormalization(axis=-1)(input1)
    xl_1 = Bidirectional(GRU(128,return_sequences=True),name='BiGRU1')(xl_0)
    drop1 = Dropout(0.5)(xl_1)
    xl_2 = Capsule(
        num_capsule=64,dim_capsule=10,
        routings=3, share_weights=True, name='Capsule1')(drop1)
    xl_3 = Flatten()(xl_2)
    #xl_4 = BatchNormalization(axis=-1)(xl_3)

    input2 = Input(shape=(kmer,39),dtype="float", name='input2')
    bio_1 = BatchNormalization(axis=-1)(input2)
    bio_2 = Bidirectional(GRU(32,return_sequences=True),name='BiGRU2')(bio_1)
    drop2 = Dropout(0.5)(bio_2)
    bio_3 = Capsule(
        num_capsule=64,dim_capsule=5,
        routings=3, share_weights=True, name='Capsule2')(drop2)
    bio_4 = Flatten()(bio_3)
    #bio_5 = BatchNormalization(axis=-1)(bio_4)

    concat = concatenate([xl_3,bio_4],axis=-1)

    norm = BatchNormalization(axis=-1,name='cont')(concat)
    #drop = Dropout(0.5)(concat)
    outputs = Dense(2,activation='softmax',name='outputs')(norm) 
    
    model = Model(inputs=[input1,input2], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='nadam',metrics=['accuracy'])
    model.summary()
    return model
    
    


