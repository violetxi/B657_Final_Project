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

# import parameters and cluster class
from homeview_params_cluster import *
from cluster import cluster

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

# load saved models. We don't need generator or the whole vae here since we only care about clustering results of latent representation
def load_models(age_group):
    # load saved models
    #vae_path = os.path.join('../models', age_group, 'homeview_ld_%d_conv_%d_id_%d_e_%d_vae.h5' % (latent_dim, num_conv, intermediate_dim, epochs))
    encoder_path = os.path.join('../models', age_group, 'homeview_ld_%d_conv_%d_id_%d_e_%d_encoder.h5' % (latent_dim, num_conv, intermediate_dim, epochs))

    #generator_path = os.path.join('../models', age_group, 'homeview_ld_%d_conv_%d_id_%d_e_%d_generator.h5' % (latent_dim, num_conv, intermediate_dim, epochs))
    #history_path = os.path.join('../models', age_group, 'homeview_ld_%d_conv_%d_id_%d_e_%d_history.pkl'  % (latent_dim, num_conv, intermediate_dim, epochs))    
    #print('vae_path is {}.\nencoder_path is {}.\ngenerator_path is {}\nhistory path is {}'.format(vae_path, generator_path, generator_path, history_path))
    #generator = keras.models.load_model(generator_path, custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})

    #vae = keras.models.load_model(vae_path, custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})
    encoder = keras.models.load_model(encoder_path, custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})
    return encoder
    # load history if saved
    '''
    fname = history_path
    try:
        with open(fname, 'rb') as fo:
            history = cPickle.load(fo)
    except:
        print "training history not saved"
    '''

# Takes a list of data points and a list of coresponding file names for clustering
# and retrieving images
def make_cluster_dataset(x, x_files):
    cluster_dataset = { x_files[i] : x[i] for i in range(len(x_files)) }

    return cluster_dataset


############################## Main ##############################
(img_source, model_source) = sys.argv[1:]
# Number of clusters
K = 2
# load dataset (testing set)
print("Loading data...")
X = data_manager(dim_x=img_cols, dim_y=img_rows, batch_size=batch_size)
x_train, x_validation, x_test = X.load_imgs(img_source, testing=True)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)


# load saved models
# Get the age group folder for saved model locations and cluster destination 
# make sure the command line input doesn't end with '/'
age_group = model_source[ model_source.rfind('/')+1 : ]
img_dest = os.path.join('../output/', img_source[img_source.rfind('/')+1 : ])
print("Age group is {}".format(age_group))
encoder = load_models(age_group)

# Cluster images and save them into corresponding cluster directories
test_files = X.testing_files
latent_representation = encoder.predict(x_test, batch_size=batch_size)
print("Latent representation: {}".format(latent_representation.shape))
x_clustering = make_cluster_dataset(latent_representation, X.testing_files)
cluster_model = cluster(x_clustering, K)
cluster_model.get_clusters()
cluster_model.save_image_clusters(img_source, img_dest)
#print(cluster_model.cluster_labels)
#print(cluster_model.file_names)
#print(cluster_model.cluster_result)

'''
# Testing with dummy data
x_dummy = np.random.normal(0, 2, (len(X.testing_files), 68))
print("Total number of datapoints: {}".format(len(X.testing_files)))
x_clustering = make_cluster_dataset(x_dummy, X.testing_files)
cluster_model = cluster(x_clustering, K)
cluster_model.get_clusters()
img_dest = os.path.join('../output/', img_source[img_source.rfind('/')+1 : ])
cluster_model.save_image_clusters(img_source, img_dest)
#print(cluster_model.cluster_labels)
#print(cluster_model.file_names)
#print(cluster_model.cluster_result)
'''
