import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cPickle

from scipy.stats import norm
from scipy.misc import imsave
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from data_manager import data_manager
import tensorflow as tf

# import parameters
from homeview_params import *

"""
loading vae model back is not a straight-forward task because of custom loss layer.
we have to define some architecture back again to specify custom loss layer and hence to load model back again.
"""
"""
This file is based on this repo: https://github.com/chaitanya100100/VAE-for-Image-Generation (models)
"""
# tensorflow or theano
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)

# encoder architecture
x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(13, 13),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(13, 13),
                padding='same', activation='relu',
                strides=(11, 11))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x

'''
vae = keras.models.load_model('../models/homeview_ld_%d_conv_%d_id_%d_e_%d_vae.h5' % (latent_dim, num_conv, intermediate_dim, epochs),
    custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})
encoder = keras.models.load_model('../models/homeview_ld_%d_conv_%d_id_%d_e_%d_encoder.h5' % (latent_dim, num_conv, intermediate_dim, epochs),
    custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})
generator = keras.models.load_model('../models/homeview_ld_%d_conv_%d_id_%d_e_%d_generator.h5' % (latent_dim, num_conv, intermediate_dim, epochs),
    custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})

# load history if saved
fname = '../models/homeview_ld_%d_conv_%d_id_%d_e_%d_history.pkl' % (latent_dim, num_conv, intermediate_dim, epochs)
try:
    with open(fname, 'rb') as fo:
        history = cPickle.load(fo)
    #print history
except:
    print "training history not saved"
'''

# load dataset to plot latent space
# load dataset
img_source = sys.argv[1]
age_group = img_source[ img_source.rfind('/')+1 : ]
print("Loading data...")

X = data_manager(dim_x=img_cols, dim_y=img_rows, batch_size=batch_size)
x_train, x_validation, x_test = X.load_imgs(img_source, testing=True)
# testing set
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

# load saved models
vae_path = os.path.join('../models', age_group, 'homeview_ld_%d_conv_%d_id_%d_e_%d_vae.h5' % (latent_dim, num_conv, intermediate_dim, epochs))                               
encoder_path = os.path.join('../models', age_group, 'homeview_ld_%d_conv_%d_id_%d_e_%d_encoder.h5' % (latent_dim, num_conv, intermediate_dim, epochs))
generator_path = os.path.join('../models', age_group, 'homeview_ld_%d_conv_%d_id_%d_e_%d_generator.h5' % (latent_dim, num_conv, intermediate_dim, epochs))
history_path = os.path.join('../models', age_group, 'homeview_ld_%d_conv_%d_id_%d_e_%d_history.pkl'  % (latent_dim, num_conv, intermediate_dim, epochs))
print('vae_path is {}.\nencoder_path is {}.\ngenerator_path is {}\nhistory path is {}'.format(vae_path, generator_path, generator_path, history_path))

generator = keras.models.load_model(generator_path, custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})
vae = keras.models.load_model(vae_path, custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})               
encoder = keras.models.load_model(encoder_path, custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})
# load history if saved
fname = history_path
try:
    with open(fname, 'rb') as fo:
        history = cPickle.load(fo)
except:
    print "training history not saved"



# Save reconstructed images with its orginal image
test_files = X.testing_files
latent_representation = encoder.predict(x_test, batch_size=batch_size)
x_decoded = generator.predict(latent_representation, batch_size=batch_size)

for i in range(len(x_decoded)):
    file_name = test_files[i]
    image_out_name = os.path.join('../output', file_name[ file_name.rfind('/')+1: ])
    print(image_out_name)
    recontructed_image = x_decoded[i]
    original_image = x_test[i]
    res = np.append(original_image, recontructed_image, axis=0)
    
    #imsave('../output/image.png', original_image)
    #imsave('../output/recon_image.png', recontructed_image)
    imsave(image_out_name, res)
