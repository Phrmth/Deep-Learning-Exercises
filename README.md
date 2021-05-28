

# What are Autoencoders

Autoencoders are a kind of artificial neural networks which is unsupervised , it tries to learn the latent features of the inputs provided and tries to regenerate the input as outputs using the latent features. In order to do so, it has an encoder block - which works to encode or create a latent feature(which is smaller in dimension) also called bottleneck layer and the decoder block regenerates the output same as the size of input. And the difference between the actual input and the generated input is called reconstruction loss and minimized over the training iterations. 

Applications - It can be used for anomaly detection, feature reduction (creating entity embeddings) and its other forms(VAE) for generative tasks like image generation, music generation etc.

This is a sample code to check the image generation using autoencoders, which has been tested on one single image, and upon 1000 iterations the model has somewhat learnt to generate back the origial image .

