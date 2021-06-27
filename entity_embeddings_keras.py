
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

# the data has all categorical independent features with 3000+ unique features if one-hot encoded and continous dependent features
# Here I am using 2 of the important features with unique values of 3000+ and 14+ 
# Creating entity embeddings for the two by training a neural network

from sklearn.preprocessing import LabelEncoder
f1_le = LabelEncoder()
f1 = f1_le.fit_transform(data.feature1)
f2_le = LabelEncoder()
f2 = f2_le.fit_transform(data.feature2)

# Embedding layer for the 'feature1' feature
n_unique_f1 = new_data['feature1'].nunique()
n_dim_f1 = int(sqrt(n_unique_f1))
input_f1 = Input(shape=(1, ))
output_f1 = Embedding(input_dim=n_unique_f1,
                        output_dim=n_dim_f1+1, name="feature1")(input_f1)
output_f1 = Reshape(target_shape=(n_dim_f1+1, ))(output_f1)


# Embedding layer for the 'feature2' feature
n_unique_f2 = new_data['feature2'].nunique()
n_dim_f2 = int(sqrt(n_unique_band))
input_f2 = Input(shape=(1, ))
output_f2 = Embedding(input_dim=n_unique_f2,
                           output_dim=n_dim_f2+1,
                           name="feature2")(input_f2)
output_f2 = Reshape(target_shape=(n_dim_f2+1,))(output_f2)



# input_layers = [input_city, input_month, input_week, input_country,input_class, input_band]
# output_layers = [output_city, output_month, output_week, output_country, output_class, output_band]
input_layers = [input_f1,  input_f2]
output_layers = [output_f1,  output_f2]
model = Concatenate()(output_layers)


model = Dense(1000, kernel_initializer="uniform")(model)
model = Activation('relu')(model)
model = Dense(500, kernel_initializer="uniform")(model)
model = Activation('relu')(model)
model = Dense(50, kernel_initializer="uniform")(model)
model = Activation('relu')(model)
model = Dense(1)(model)
# model = BatchNormalization()
# model = Activation('sigmoid')(model)
# And finally our output layer
# model = Dense(1)(model)

# Put it all together and compile the model
model = Model(inputs=input_layers, outputs=model)
model.summary()
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])


hist = model.fit([f1,f2], output, epochs =  30,batch_size=10000, verbose= 1, validation_split=.2 )

f1_embed = model.get_layer('feature1').get_weights()[0]
f2_embed = model.get_layer('feature2').get_weights()[0]


# Clustering on learnt entity embeddings, 
# code from https://github.com/entron/entity-embedding-rossmann/blob/master/plot_embeddings.ipynb

from sklearn import manifold
tsne = manifold.TSNE(init='pca', random_state=0, method='exact', perplexity=5, learning_rate=100,n_jobs=-1)
Y = tsne.fit_transform(f1_embed)
names = f1_le.classes_

plt.figure(figsize=(80,80))
plt.scatter(-Y[:, 0], -Y[:, 1], c = pred)
for i, txt in enumerate(names):
    plt.annotate(txt, (-Y[i, 0],-Y[i, 1]), xytext = (1, 10), textcoords = 'offset points')

