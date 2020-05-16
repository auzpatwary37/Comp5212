#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GCN + Feed Forward neural Network Model
Feel free to tweak and play with the parameters
And let me know which one works best. 
All the best!!! 
"""
"""
Created on Mon May 11 19:58:46 2020

@author: ashraf
"""
import pandas as pd
import numpy as np
from scipy import sparse
import keras.backend as k



from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import LeakyReLU as LRelu

from tensorflow.keras.layers import ReLU 

from tensorflow.keras.layers import Dropout



from spektral.layers import GraphConv
from spektral.utils.convolution import localpooling_filter

from matplotlib import pyplot as plt


#read and process the time stamps on the lyft_data table
lyft_data = pd.read_csv("LYFT_Manhattan_data.csv",index_col=0)
lyft_data['Pickup_DateTime'] = pd.to_datetime(lyft_data['Pickup_DateTime'])
lyft_data['DropOff_datetime'] = pd.to_datetime(lyft_data['DropOff_datetime'])
lyft_data['Pickup_DateTime'] = lyft_data['Pickup_DateTime'].apply(lambda x: x.timestamp( ))
lyft_data['DropOff_datetime'] = lyft_data['DropOff_datetime'].apply(lambda x: x.timestamp( ))


#read area data
area_stat = pd.read_csv("areaStat.csv")


#remove the area id and save the features as area_stat_np
area_stat_np = area_stat.loc[:, area_stat.columns != 'Matching_area'].to_numpy()

#read the adjacent matrix 
area_A = pd.read_csv("adjacent_matrix_Manhattan.csv",index_col=0).to_numpy().astype(np.float32)

#convert the adjacent matrix to sparse (may be not nencessary)
A = sparse.csr_matrix(area_A)

    

# Parameters

K = 2                   # Degree of propagation (Parameter of the GCN, do not know what it means though)
N = area_stat.shape[0]          # Number of nodes in the graph (Do not change)
F = area_stat_np.shape[1]          # Original size of node features (Do not change)
l2_reg = 5e-6           # L2 regularization rate
learning_rate = 0.005     # Learning rate
epochs = 200         # Number of training epochs
embedding_vecor_length1 = 64
embedding_vecor_length2 = 32
include_batch_norm_layers = True
include_dropout_layers = False


# create the trip data set and devide it in test and train
def create_data(d,frac,random_state):
    d = d.sample(frac=1)#random shuffle
    one_hot_o = pd.get_dummies(d['PUlocationID'], prefix='o')#get the one hot area lables for pick up loacation
    one_hot_d = pd.get_dummies(d['DOlocationID'], prefix='d')#get the one hot area lables for drop off loacation
    label_flag = d.get('SR_Flag')#get the trip type flag
    d = d.drop('SR_Flag',axis = 1)#drop the flag from the original data frame
    d = d.drop('PUlocationID',axis = 1)#drop the area label for pickup location
    d = d.drop('DOlocationID',axis = 1)#drop the area label for dropoff location
    start_o = d.shape[1]#save the starting index of the pickup area's one hot features
    d = d.join(one_hot_o)#add the one hot feature vector for the pickup area
    end_o = d.shape[1]-1#save the ending index of the pickup area's one hot features
    start_d = end_o+1#save the starting index of the dropoff area's one hot features
    d = d.join(one_hot_d)#add the one hot feature vector for the dropoff area
    d = d.join(label_flag)#add the trip flag back
    
    train_d = d.sample(frac = frac,random_state=random_state)#divide the dataframe into train and test
    test_d = d.drop(train_d.index)#divide the dataframe into train and test
    label_train = pd.get_dummies(train_d['SR_Flag'])#convert the trip flag into one hot
    label_test = pd.get_dummies(test_d['SR_Flag'])#convert the trip flag into one hot
    train_d = train_d.drop('SR_Flag',axis = 1)#drop the original trip flag
    test_d = test_d.drop('SR_Flag',axis = 1)#drop the original trip flag
    
    #return the test and train data sets including their lables in numpy float array format
    d_tr = train_d.to_numpy().astype(np.float32)
    d_tst = test_d.to_numpy().astype(np.float32)
    l_tr = label_train.to_numpy().astype(np.float32)
    l_ts = label_test.to_numpy().astype(np.float32)
    
    return [d_tr,l_tr,d_tst,l_ts,start_o,end_o,start_d]
    
[train_np,label_train_np,test_np, label_test_np,start_o,end_o,start_d] = create_data(lyft_data,0.8001,200)

#save the trip data feature vector size
trip_feature = train_np.shape[1]


# Preprocessing operations (Do not know what this does. Coppied from the GCN example provided by Speckter)
fltr = localpooling_filter(A).astype('f4')
for i in range(K - 1):
    fltr = fltr.dot(fltr)
fltr.sort_indices()



# Model definition

#GCN input 
area_in = Input(shape=(F, ))#area feature 
fltr_in = Input((N, ), sparse=True)#adjacent matrix

bn0 = BN(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', renorm_momentum=0.99,   
)(area_in)#batch norm on the area data

if not include_batch_norm_layers: 
    bn0 = area_in

g1 = GraphConv(embedding_vecor_length1, #first graph conv layer
                   kernel_regularizer=l2(l2_reg),
                   use_bias=True)([bn0, fltr_in])
bn1 = BN(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', renorm_momentum=0.99,   
)(g1)#bathc norm on g1
    
if not include_batch_norm_layers: 
    bn1 = g1

ac1 = LRelu(alpha=0.3)(bn1)#activation on g1 (Chosen LickyReLU)


g2 = GraphConv(embedding_vecor_length2, # Second graph conv layer
                   kernel_regularizer=l2(l2_reg),
                   use_bias=True)([ac1, fltr_in])

bn2 = BN(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', renorm_momentum=0.99,   
)(g2)#batch norm on g2

if not include_batch_norm_layers: 
    bn2 = g2
    
ac2 = LRelu(alpha=0.3)(bn2)#activation on g2

def gcn_to_trip(x):#lambda function corresponding to the lambda layer
    trip_input=x[0]
    graph_output=x[1]
    o = k.dot(trip_input[:,start_o:end_o+1],graph_output)#get area encoding from g2 corresponding to the pickup in trip data 
    d = k.dot(trip_input[:,start_d:],graph_output)#get area encoding from g2 corresponding to the dropoff in trip data
    t = trip_input[:,:start_o]#Get the actual trip features
    out = k.concatenate((t,o,d),axis=1)#Merge trip, pickup and dropoff features

    return out

trip_in = Input(shape = (trip_feature, ))#trip feature input


lam1 = Lambda(gcn_to_trip)([trip_in,g2])#The lambda layer to apply the gcn_to_trip function


bn3 = BN(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', renorm_momentum=0.99,   
)(lam1)#batchnorm on trip features 

if not include_batch_norm_layers: 
    bn3 = lam1

d1 = Dense(units=500,input_dim=trip_feature+embedding_vecor_length2*2,
            kernel_regularizer=l2(l2_reg),
            use_bias=True)(lam1)#first dense layer

bn4 = BN(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', renorm_momentum=0.99,   
)(d1)#batch norm on d1

if not include_batch_norm_layers: 
    bn4 = d1

ac3 = ReLU()(bn4)#activation on d1

dr1 = Dropout(rate = .2)(ac3)#drop off on d1


if not include_dropout_layers: 
    dr1 = ac3


d2 = Dense(units=500,
            kernel_regularizer=l2(l2_reg),
            use_bias=True)(dr1)#Second dense layer 

bn5 = BN(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones', renorm_momentum=0.99,   
)(d2)#batch norm on d2

if not include_batch_norm_layers: 
    bn5 = d2

ac4 = ReLU()(bn5)#activation on d2

dr2 = Dropout(rate = .2)(ac4)#dropoff on d2

if not include_dropout_layers: 
    dr2 = ac4

output = Dense(units=2,kernel_regularizer=l2(l2_reg),activation='softmax',use_bias=True)(dr2)#output softmax layer


model = Model(inputs=[area_in, fltr_in, trip_in], outputs=output)#Build the model

optimizer = Adam(lr=learning_rate)#Set the optimizer
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',metrics=['accuracy'])#Compile the model
model.summary()#Print model summary

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)#plot model as block diagram

def generator(train_np, label_train_np):#data generator to feed the fitting method with dataset per batch

  samples_per_epoch = train_np.shape[0]
  number_of_batches = samples_per_epoch/N
  counter=0

  while 1:

    X_batch = [area_stat_np,fltr.todense(),train_np[N*counter:N*(counter+1)]]
    y_batch = label_train_np[N*counter:N*(counter+1)]
    counter += 1
    yield X_batch,y_batch

    #restart counter to yeild data in the next epoch as well
    if counter >= number_of_batches:
        counter = 0

# model.fit([area_stat_np,fltr,train_np], label_train_np, epochs=3, batch_size=63)

history  = model.fit_generator(#fit the model
    generator(train_np,label_train_np),
    validation_data = generator(test_np,label_test_np),
    validation_steps = test_np.shape[0]/N,
    epochs=epochs,
    steps_per_epoch = int(train_np.shape[0]/N)
)

#Plot training history
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy.png', dpi=300)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss.png', dpi=300)




