
'''
In this section we would be creating a neural network to learn entity embeddings of the categorical variables passed to it . The dependent variable or output 
variable is the one against which the entity embeddings needs to be created . The embeddings can be further clustered using kmeans to identify possible clusters 
in the categorical variable. This can be very useful if we have a lot of categories and onehot encoding is not the right approach . 

'''

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, Activation, BachNormalization
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers.merge import Concatenate
from numpy import sqrt

