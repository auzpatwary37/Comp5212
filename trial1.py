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


from spektral.datasets import citation
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
F = area_stat.shape[1]          # Original size of node features
l2_reg = 5e-6           # L2 regularization rate
learning_rate = 0.2     # Learning rate
epochs = 2000         # Number of training epochs
embedding_vecor_length1 = 32
embedding_vecor_length2 = 32



# Get one hot encoding of columns B
one_hot_o_train = pd.get_dummies(train['PUlocationID'], prefix='o')
one_hot_d_train = pd.get_dummies(train['DOlocationID'], prefix='d')

one_hot_o_test = pd.get_dummies(test['PUlocationID'], prefix='o')
one_hot_d_test = pd.get_dummies(test['DOlocationID'], prefix='d')

label_train = pd.get_dummies(train['SR_Flag'])
label_train_np = label_train.to_numpy().astype(np.float32)

label_test = pd.get_dummies(test['SR_Flag'])
label_test_np = label_test.to_numpy().astype(np.float32)

train = train.drop('SR_Flag',axis = 1)
test = test.drop('SR_Flag',axis = 1)

# Drop column B as it is now encoded
train = train.drop('PUlocationID',axis = 1)
test = test.drop('PUlocationID',axis = 1)
train = train.drop('DOlocationID',axis = 1)
test = test.drop('DOlocationID',axis = 1)
start_o = lyft_data.shape[1]

# Join the encoded df
train = train.join(one_hot_o_train)
test = test.join(one_hot_o_test)
end_o = train.shape[1]-1
start_d = end_o+1

train = train.join(one_hot_d_train)
train_np = train.to_numpy().astype(np.float32)
test = test.join(one_hot_d_test)
test_np = test.to_numpy().astype(np.float32)

trip_feature = lyft_data.shape[1]


# Preprocessing operations
fltr = localpooling_filter(A).astype('f4')
for i in range(K - 1):
    fltr = fltr.dot(fltr)
fltr.sort_indices()


area_in= Lambda(lambda:area_stat_np, name="lambda_layer_area")
fltr_in = Lambda(lambda:fltr, name="lambda_layer_fltr")

# Model definition
# area_in = Input(shape=(F, ))
# fltr_in = Input((N, ), sparse=True)
g1 = GraphConv(embedding_vecor_length1, #first graph conv layer
                   activation='relu',
                   kernel_regularizer=l2(l2_reg),
                   use_bias=False)([area_in, fltr_in])
g2 = GraphConv(N, # Second graph conv layer
                   activation='softmax',
                   kernel_regularizer=l2(l2_reg),
                   use_bias=False)([g1, fltr_in])

model = Model(inputs=[area_in,fltr_in], outputs=[g2])

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


# def gcn_to_trip(x):
#     trip_input=x[0]
#     graph_output=x[1]
#     o = k.transpose(k.dot(k.transpose(trip_input[:,start_o:end_o]),k.transpose(graph_output)))
#     d = k.transpose(k.dot(k.transpose(trip_input[:,start_d:]),k.transpose(graph_output)))
#     t = trip_input[:,:start_o]
#     out = k.concatenate((t,o,d),axis=-1)
#     return out


# trip_in = Input(shape = (trip_feature, ))

# l1 = Lambda(gcn_to_trip, name="lambda_layer")([trip_in,g2])

# d1 = Dense(units=500,
#            activation='relu',
#            kernel_regularizer=l2(l2_reg),
#            use_bias=True)(l1)

# d2 = Dense(units=500,
#            activation='relu',
#            kernel_regularizer=l2(l2_reg),
#            use_bias=True)(d1)

# output = Dense(units=2,
#            activation='softmax',
#            kernel_regularizer=l2(l2_reg),
#            use_bias=True)(d2)

# model = Model(inputs=[area_in, fltr_in, trip_in], outputs=output)

# optimizer = Adam(lr=learning_rate)
# model.compile(optimizer=optimizer,
#               loss='categorical_crossentropy',
#               weighted_metrics=['acc'])
# model.summary()

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



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


