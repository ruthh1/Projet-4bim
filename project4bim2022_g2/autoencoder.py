'''
This module implements a variational autoencoder for the celeba dataset
'''

from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, UpSampling2D, Reshape
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# custom import
from project4bim2022_g2.utils import *

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors

plt.rcParams['figure.figsize'] = (10, 6)

'''
Mixture of Gaussians distribution
'''


def get_prior(num_modes, latent_dim):
    """
    Defines a prior distribution that is a mixture of Gaussians. 
    This is a more flexible distribution that is comprised of $K$ separate Gaussians, 
    that are combined together with some weighting assigned to each.


    Args:
        num_modes (int): 
        latent_dim (int): latent dimension we want to encode the image in

    Returns:
        tfp.distributions.MixtureSameFamily : Mixture of gaussian distribution, with variable means and 
        standart deviations
    """
    prior = tfd.MixtureSameFamily(
        # The mixture distribution is a list of probability for the distributions. Here it is set to be
        # all equal
        mixture_distribution=tfd.Categorical(
            probs=[1 / num_modes, ] * num_modes),
        # The components distribution is made of two lists : loc represents the mean of the distribution,
        # or the "location" of the distribution.
        # scale_diag represents the
        components_distribution=tfd.MultivariateNormalDiag(
            loc=tf.Variable(tf.random.normal(shape=[num_modes, latent_dim])),
            scale_diag=tfp.util.TransformedVariable(tf.Variable(
                tf.ones(shape=[num_modes, latent_dim])), bijector=tfb.Softplus())
        )
    )
    return prior


'''
Define the encoder Network
'''


def get_kl_regularizer(prior_distribution):
    """The kl regularizer is a part of the loss function. It takes the prior distribution as an
    input will be used as the activity_regularizer of the tfpl.MultivariateNormalTriL layer of the encoder.

    Args:
        prior_distribution (tfp.distributions.MixtureSameFamily): 

    Returns:
        tensorflow_probability.python.layers.distribution_layer.KLDivergenceRegularizer :
    """
    divergence_regularizer = tfpl.KLDivergenceRegularizer(
        prior_distribution,
        use_exact_kl=False,
        weight=1.0,
        test_points_fn=lambda q: q.sample(3),
        test_points_reduce_axis=(0, 1)
    )
    return divergence_regularizer


def get_encoder(latent_dim, kl_regularizer):
    """Generates an encoder with a given latent dimension and a kl regularizer

    Args:
        latent_dim (int): dimension of the latent space we want to encode the image to
        kl_regularizer (tensorflow_probability.python.layers.distribution_layer.KLDivergenceRegularizer): kl regularizer

    Returns:
        keras.engine.sequential.Sequential : encoder with all the layers we want
    """
    encoder = Sequential([
        Conv2D(32, (4, 4), activation='relu', strides=2,
               padding='SAME', input_shape=(64, 64, 3)),
        BatchNormalization(),
        Conv2D(64, (4, 4), activation='relu', strides=2, padding='SAME'),
        BatchNormalization(),
        Conv2D(128, (4, 4), activation='relu', strides=2, padding='SAME'),
        BatchNormalization(),
        Conv2D(256, (4, 4), activation='relu', strides=2, padding='SAME'),
        BatchNormalization(),
        Flatten(),
        Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim)),
        tfpl.MultivariateNormalTriL(
            latent_dim, activity_regularizer=kl_regularizer)
    ])
    return encoder


'''
Define the decoder network
'''


def get_decoder(latent_dim):
    """Generates the decoder of the vae, with a given latent dimension

    Args:
        latent_dim (int): latent dimension the image is encoded into

    Returns:
        keras.engine.sequential.Sequential: decoder of the vae model
    """
    decoder = Sequential([
        Dense(4096, activation='relu', input_shape=(latent_dim, )),
        Reshape((4, 4, 256)),
        UpSampling2D(size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='SAME'),
        UpSampling2D(size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='SAME'),
        UpSampling2D(size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='SAME'),
        UpSampling2D(size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='SAME'),
        Conv2D(3, (3, 3), padding='SAME'),
        Flatten(),
        tfpl.IndependentBernoulli(event_shape=(64, 64, 3))
    ])
    return decoder


'''
Define the average reconstruction loss
'''


def reconstruction_loss(batch_of_images, decoding_dist):
    """
    The function takes batch_of_images (Tensor containing a batch of input images to
    the encoder) and decoding_dist (output distribution of decoder after passing the 
    image batch through the encoder and decoder) as arguments.
    The function should return the scalar average expected reconstruction loss.

    Returns:
        tf.Tensor : scalar average expected reconstruction loss
    """
    return -tf.reduce_mean(decoding_dist.log_prob(batch_of_images), axis=0)


def reconstruct(encoder, decoder, batch_of_images):
    """
    The function takes the encoder, decoder and batch_of_images as inputs, which
    should be used to compute the reconstructions.
    The function should then return the reconstructions Tensor.

    Args:
        encoder (Sequential): encoder of the model
        decoder (Sequential): decoder of the model
        batch_of_images (ndarray): list of images we want to reconstruct

    Returns:
        tf.Tensor : reconstruction tensor
    """
    approx_posterior = encoder(batch_of_images)
    decoding_dist = decoder(approx_posterior.mean())
    return decoding_dist.mean()


def plot_reconstructions_test_ds(encoder, decoder):
    """
    Plots the reconstruction of the images versus the initial images

    Args:
        encoder (Sequential): encoder of the vae
        decoder (Sequential): decoder of the vae
    """
    n_reconstructions = 7
    num_test_files = np.load('./img_align_celeba/dataset_test.npy').shape[0]
    test_ds_for_reconstructions = load_dataset('dataset_test')
    # Run your function to compute reconstructions of random samples from the test dataset
    for all_test_images, _ in test_ds_for_reconstructions.batch(num_test_files).take(1):
        all_test_images_np = all_test_images.numpy()
    example_images = all_test_images_np[np.random.choice(
        num_test_files, n_reconstructions, replace=False)]

    reconstructions = reconstruct(encoder, decoder, example_images).numpy()

    # Plot the reconstructions

    f, axs = plt.subplots(2, n_reconstructions, figsize=(16, 6))
    axs[0, n_reconstructions // 2].set_title("Original test images")
    axs[1, n_reconstructions // 2].set_title("Reconstructed images")
    for j in range(n_reconstructions):
        axs[0, j].imshow(example_images[j])
        axs[1, j].imshow(reconstructions[j])
        axs[0, j].axis('off')
        axs[1, j].axis('off')

    plt.tight_layout()
    plt.show()

    return None


'''
Sample new images from the generative model
'''


def generate_images(prior, decoder, n_samples):
    """
    The function takes the prior distribution, decoder and number of samples as inputs, which
    should be used to generate the images.
    The function should then return the batch of generated images.

    Args:
        prior (MixtureSameFamily): prior distribution
        decoder (Sequential): trained decoder of the vae
        n_samples (int): targeted number of samples

    Returns:
        tf.Tensor : images generated
    """
    z = prior.sample(n_samples)
    sampled_images = decoder(z).mean()
    for count, value in enumerate(sampled_images):
        image = Image.fromarray((value.numpy()*255).astype('uint8'), 'RGB')
        image.save('gen'+str(count)+'.png')
    return sampled_images


def generate_latent_vectors(prior, n_samples):
    ''''''
    Z = prior.sample(n_samples)
    Z2 = Z.numpy()
    '''if n_samples == 1:
        Z3 = []
        for element in Z2[0]:
            Z3.append(element)
        Z2 = np.array(Z3)'''
    return Z2


def reconstruct_image_from_latent_vectors(decoder, Z):
    ''''''
    return decoder(Z).mean()


def plot_recontructed_images(X):
    ''''''
    n_samples = len(X)
    f, axs = plt.subplots(1, n_samples, figsize=(16, 6))
    if n_samples == 1:
        plt.imshow(X[0])
    else:
        for j in range(n_samples):
            axs[j].imshow(X[j])
            axs[j].axis('off')
    plt.show()


def plot_generate_images(prior, decoder):
    '''
    The function generates a list of images, plots them and saves them

    Args:
        prior (MixtureSameFamily): prior distribution
        decoder (Sequential): trained decoder of the vae
    '''
    n_samples = 10
    sampled_images = generate_images(prior, decoder, n_samples)

    f, axs = plt.subplots(1, n_samples, figsize=(16, 6))

    for j in range(n_samples):
        axs[j].imshow(sampled_images[j])
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('image_generation.png')


def train(vae, train_ds, val_ds, epochs):
    vae.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return None

##========================= MAIN =======================##


if __name__ == "__main__":
    print("Tensorflow Version: ", tf.__version__)
    print("Tensorflow Probability Version: ", tfp.__version__)

    #train_ds, val_ds, test_ds = load_datasets()

    # Run your function to get the prior distribution with 2 components and latent_dim = 50
    prior = get_prior(num_modes=2, latent_dim=50)

    # Run your function to get the encoder
    encoder = get_encoder(
        latent_dim=50, kl_regularizer=get_kl_regularizer(prior))

    # Run your function to get the decoder
    decoder = get_decoder(latent_dim=50)

    # ## Link the encoder and decoder together
    #
    # Connects `encoder` and `decoder` to form the end-to-end architecture.
    vae = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    # ## Compile and fit the model
    #
    # It's now time to compile and train the model. Note that, it is recommand to use Hardware accelerator while training.

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    vae.compile(optimizer=optimizer, loss=reconstruction_loss)

    # Train the model
    #vae.fit(train_ds, validation_data=val_ds, epochs=1)
    # Or just load the weights
    encoder.load_weights("./model_vae/encoder/saved_encoder")
    decoder.load_weights("./model_vae/decoder/saved_decoder")

    #vae.fit(train_ds, validation_data=val_ds, epochs=1)

    # encoder.save_weights("./model_vae/encoder/saved_encoder")
    # decoder.save_weights("./model_vae/decoder/saved_decoder")

    # Evaluate the model on the test set
    #test_loss = vae.evaluate(test_ds)
    #print("Test loss: {}".format(test_loss))

    #plot_reconstructions_test_ds(encoder, decoder)

    plot_generate_images(prior, decoder)
