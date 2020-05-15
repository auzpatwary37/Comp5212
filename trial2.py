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

from spektral.datasets import citation
from spektral.layers import GraphConv
from spektral.utils.convolution import localpooling_filter



lyft_data = pd.read_csv("LYFT_Manhattan_data.csv",index_col=0)

area_stat = pd.read_csv("areaStat.csv")

area_stat_np = area_stat.loc[:, area_stat.columns != 'Matching_area'].to_numpy()
area_A = pd.read_csv("adjacent_matrix_Manhattan.csv",index_col=0).to_numpy()

A = sparse.csr_matrix(area_A)

# Get one hot encoding of columns B
one_hot_o = pd.get_dummies(lyft_data['PUlocationID'], prefix='o')
one_hot_d = pd.get_dummies(lyft_data['DOlocationID'], prefix='d')
trip_label = pd.get_dummies(lyft_data['SR_Flag'])
# Drop column B as it is now encoded
lyft_data = lyft_data.drop('PUlocationID',axis = 1)
lyft_data = lyft_data.drop('DOlocationID',axis = 1)
lyft_data = lyft_data.drop('SR_Flag',axis = 1)

start_o = lyft_data.shape[1]

# Join the encoded df
lyft_data = lyft_data.join(one_hot_o)
end_o = lyft_data.shape[1]-1
start_d = end_o+1

lyft_data = lyft_data.join(one_hot_d)
l = lyft_data.to_numpy()

o = l[:,start_o:end_o]
d = l[:,start_d:]
t = l[:,:start_o]

t1 = tf.keras.backend.random_uniform([10,100])

def gcn_to_trip(x):
    trip_input=x[0]
    graph_output=x[1]
    o = k.transpose(k.dot(k.transpose(trip_input[:,start_o:end_o]),k.transpose(graph_output)))
    d = k.transpose(k.dot(k.transpose(trip_input[:,start_d:]),k.transpose(graph_output)))
    t = trip_input[:,:start_o]
    out = k.concatenate((t,o,d),axis=-1)
    return out
    

