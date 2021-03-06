import os
import sys

import numpy as np
import tensorflow as tf

'''
This script imports images from given datasets and turn them into usable
input X for tensorflow framework. 
The script is based on Tensorflow's tutporial (https://www.tensorflow.org/programmers_guide/datasets#decoding_image_data_and_resizing_it)
'''
class data_manager():
    
    def __init__(self, dim_x, dim_y, training=0.7, validation=0.1, testing=0.2, batch_size=50):
        ''' dim_x - width and dim_y - height default to 640x480
        training, validation and testing are the percentage of files used for the beta VAE
        '''
        self.dim_x, self.dim_y = dim_x, dim_y
        self.training_portion = training
        self.validation_portion = validation
        self.testing_portion = testing
        self.batch_size = batch_size
        

    # Parse every image in the dataset using 'map'
    def _parse_function(self, filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize_images(image_decoded, [ self.dim_y, self.dim_x ])

        return image

    # Load images from source path into a tf Dataset object (no labels)
    def load_imgs(self, img_source, testing=False):
        # Get all the file names
        file_names = [ os.path.join(img_source, f) for f in os.listdir(img_source) if f.endswith('.jpg') ]
        
        # Split datasets into training, validation and testing sets
        total_samples = len(file_names)
        training_samples = file_names[ : int(total_samples * self.training_portion) ]
        validation_samples = file_names[ int(total_samples*self.training_portion) : int(total_samples * self.training_portion) + int(total_samples*self.validation_portion) ]
        testing_samples = file_names[ int(total_samples * self.training_portion + total_samples*self.validation_portion) : ]
        # Size of training, validation and testing samples
        n_training, n_validation, n_testing = len(training_samples), len(validation_samples), len(testing_samples)
        # Start a tf session
        sess = tf.Session()
        # This is later used for testing and drawing the corresponding reconstructed images
        self.testing_files = testing_samples

        # Create a dataset returning slices of 'filenames' and create iterator for training, validation and testing
        # training and validation
        if not testing:
            print("Preparing testing and validation set")
            training_filenames = tf.constant(training_samples)
            validation_filenames = tf.constant(validation_samples)
            training_dataset = tf.data.Dataset.from_tensor_slices((training_filenames))
            validation_dataset = tf.data.Dataset.from_tensor_slices((validation_filenames))        
            training_dataset = training_dataset.map(self._parse_function)
            training_dataset = training_dataset.batch(n_training)        
            validation_dataset = validation_dataset.map(self._parse_function)
            validation_dataset = validation_dataset.batch(n_validation)
            train = training_dataset.make_initializable_iterator()
            validation = validation_dataset.make_initializable_iterator()
            test = None
            print("Initializing training set..")
            sess.run(train.initializer)
            print("Initializing validation set..")
            sess.run(validation.initializer)
            # return np-array like image data
            train = sess.run(train.get_next())
            print("Finalized training set..")
            validation = sess.run(validation.get_next())
            print("Finalized validation set..")
        # testing 
        else:
            print("Preparing testing set..")
            train, validation = None, None
            testing_filenames = tf.constant(testing_samples)
            testing_dataset = tf.data.Dataset.from_tensor_slices((testing_filenames))
            testing_dataset = testing_dataset.map(self._parse_function)
            testing_dataset = testing_dataset.batch(n_testing)
            test = testing_dataset.make_initializable_iterator()
            print("Initializing testing set..")
            sess.run(test.initializer)
            test = sess.run(test.get_next())
            print("Finalized testing set..")    

        return train, validation, test
