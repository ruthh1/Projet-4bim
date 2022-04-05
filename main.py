from autoencoder import *
#from Traitement_Database import *
#from utils import *
from Genetic_algorithm import *

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#import tensorflow as tf



#On construit l'encodeur et le décodeur et on charge les poids calculés lors de l'entrainement du réseau de neurones
prior = get_prior(num_modes=2, latent_dim=50)
encoder = get_encoder(latent_dim=50, kl_regularizer=get_kl_regularizer(prior))
decoder = get_decoder(latent_dim=50)

#encoder.load_weights("VAE_encoder2")
#decoder.load_weights("VAE_decoder2")

encoder.load_weights("./model_vae/encoder/saved_encoder")
decoder.load_weights("./model_vae/decoder/saved_decoder")



#

z = generate_latent_vectors(prior, 10)
z2 = reconstruct_image_from_latent_vectors(decoder, z)
