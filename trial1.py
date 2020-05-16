#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a test for the dataset read
"""
"""
Created on Mon May 11 19:58:46 2020

@author: ashraf
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import sparse
import keras.backend as k

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model



from spektral.layers import GraphConv
from spektral.utils.convolution import localpooling_filter


lyft_data = pd.read_csv("LYFT_Manhattan_data.csv",index_col=0)
lyft_data['Pickup_DateTime'] = pd.to_datetime(lyft_data['Pickup_DateTime'])
lyft_data['DropOff_datetime'] = pd.to_datetime(lyft_data['DropOff_datetime'])
lyft_data['Pickup_DateTime'] = lyft_data['Pickup_DateTime'].apply(lambda x: x.timestamp( ))
lyft_data['DropOff_datetime'] = lyft_data['DropOff_datetime'].apply(lambda x: x.timestamp( ))

train=lyft_data.sample(frac=0.8,random_state=200) #random state is a seed value
test=lyft_data.drop(train.index)

area_stat = pd.read_csv("areaStat.csv")



#print(lyft_data.describe())

#lyft_sample = lyft_data.sample(1000)

#areas = area_stat.duplicated("Matching_area")#check if there is duplicate 
area_stat_np = area_stat.loc[:, area_stat.columns != 'Matching_area'].to_numpy()
area_A = pd.read_csv("adjacent_matrix_Manhattan.csv",index_col=0).to_numpy().astype(np.float32)

A = sparse.csr_matrix(area_A)

    

# Parameters
K = 2                   # Degree of propagation
N = area_stat.shape[0]          # Number of nodes in the graph
F = area_stat_np.shape[1]          # Original size of node features
l2_reg = 5e-6           # L2 regularization rate
learning_rate = 0.2     # Learning rate
epochs = 2000         # Number of training epochs
embedding_vecor_length1 = 32
embedding_vecor_length2 = 32



# Get one hot encoding of columns B
def create_data(d):
    one_hot_o = pd.get_dummies(d['PUlocationID'], prefix='o')
    one_hot_d = pd.get_dummies(d['DOlocationID'], prefix='d')
    label = pd.get_dummies(d['SR_Flag'])
    label_np = label.to_numpy().astype(np.float32)
    d = d.drop('SR_Flag',axis = 1)
    d = d.drop('PUlocationID',axis = 1)
    d = d.drop('DOlocationID',axis = 1)
    start_o = d.shape[1]
    d = d.join(one_hot_o)
    end_o = d.shape[1]-1
    start_d = end_o+1
    d = d.join(one_hot_d)
    
    d_np = d.to_numpy().astype(np.float32)
    return [d_np,label_np,start_o,end_o,start_d]
    
[train_np,label_train_np,start_o,end_o,start_d] = create_data(train)
[test_np,label_test_np,start_o1,end_o1,start_d1] = create_data(test)




trip_feature = train_np.shape[1]


# Preprocessing operations
fltr = localpooling_filter(A).astype('f4')
for i in range(K - 1):
    fltr = fltr.dot(fltr)
fltr.sort_indices()


# area_in= Lambda(lambda:area_stat_np, name="lambda_layer_area")
# fltr_in = Lambda(lambda:fltr, name="lambda_layer_fltr")

# Model definition
area_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)
g1 = GraphConv(embedding_vecor_length1, #first graph conv layer
                   activation='relu',
                   kernel_regularizer=l2(l2_reg),
                   use_bias=False)([area_in, fltr_in])
g2 = GraphConv(embedding_vecor_length2, # Second graph conv layer
                   activation='softmax',
                   kernel_regularizer=l2(l2_reg),
                   use_bias=False)([g1, fltr_in])

# model = Model(inputs=[area_in,fltr_in], outputs=[g2])

#print(model.output_shape)

# gcn = Model(inputs=[area_in, fltr_in], outputs=g2)
# optimizer = Adam(lr=learning_rate)
# gcn.compile(optimizer=optimizer,
#               loss='categorical_crossentropy',
#               weighted_metrics=['acc'])
# gcn.summary()

# # Train model
# validation_data = ([X, fltr], y, val_mask)
# model.fit([X, fltr],
#           y,
#           sample_weight=train_mask,
#           epochs=epochs,
#           batch_size=N,
#           validation_data=validation_data,
#           shuffle=False,  # Shuffling data means shuffling the whole graph
#           callbacks=[
#               EarlyStopping(patience=es_patience,  restore_best_weights=True)
#           ])

# # Evaluate model
# print('Evaluating model.')
# eval_results = model.evaluate([X, fltr],
#                               y,
#                               sample_weight=test_mask,
#                               batch_size=N)


def gcn_to_trip(x):
    trip_input=x[0]
    graph_output=x[1]
    o = k.dot(trip_input[:,start_o:end_o+1],graph_output)
    d = k.dot(trip_input[:,start_d:],graph_output)
    t = trip_input[:,:start_o]
    out = k.concatenate((t,o,d),axis=1)
    
    return out




trip_in = Input(shape = (trip_feature, ))

lam1 = Lambda(gcn_to_trip)([trip_in,g2])

d1 = Dense(units=500,input_dim=trip_feature+embedding_vecor_length2*2,
            activation='relu',
            kernel_regularizer=l2(l2_reg),
            use_bias=True)(lam1)

d2 = Dense(units=500,
            activation='relu',
            kernel_regularizer=l2(l2_reg),
            use_bias=True)(d1)

output = Dense(units=2,activation='softmax')(d2)

model = Model(inputs=[area_in, fltr_in, trip_in], outputs=output)

optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



# model.fit([area_stat_np,fltr,train_np], label_train_np, epochs=3, batch_size=63)
model.train_on_batch([area_stat_np,fltr,train_np[:63,:]], label_train_np[:63,:])

# validation_data = ([area_stat_np,fltr,test_np], label_test_np)

# model.fit([area_stat_np,fltr,train_np],
#           label_train_np,
#           epochs=epochs,
#           batch_size=N,
#           validation_data=validation_data,
#           shuffle=False,  # Shuffling data means shuffling the whole graph
#           )

"""
model = Sequential()

x = input() 

model.add(Embedding(input_size, embedding_vecor_length, input_size))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, nb_epoch=3, batch_size=64)


"""


