import sys
import os
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imsave
from shutil import copyfile

class cluster():
    # Constructor takes a hash of image names and their corresponding latent representation
    # and number of clusters
    def __init__(self, z_with_fname, K):
        self.data_dict = z_with_fname
        self.data_set = np.array(z_with_fname.values())
        self.file_names = z_with_fname.keys()
        self.K = K
        
    # Takes in an array of data points and get cluster labels and centers (K-means)
    # cluster_result is a hash of image file names and the cluster they belong to
    def get_clusters(self):
        kmeans = KMeans(n_clusters=self.K, random_state=0, max_iter=5000).fit(self.data_set)
        self.cluster_labels = kmeans.labels_
        
        self.cluster_result = { self.file_names[i] : kmeans.labels_[i]  for i in range(len(self.file_names)) }

    # Save images to coresponding clusters to a given location based on the result
    # dest will be in a format of ../output/age_group
    def save_image_clusters(self, img_source, dest):
        all_clusters, cluster_count = np.unique(self.cluster_labels, return_counts=True)
        print("Data points in clusters:", dict(zip(all_clusters, cluster_count)))
        # Create directories for each cluster { cluster # : image destination location }
        cluster_img_dest = { cluster : os.path.join(dest, "cluster"+str(cluster)) for cluster in all_clusters }
        for c in cluster_img_dest:
            try:
                os.mkdir(cluster_img_dest[c])
            except:
                pass

        # Remove existing files in the clustering folders to prevent confusion from previous results
        for c in cluster_img_dest:
            try:
                sh.rm(sh.glob(os.path.join(cluster_img_dest[c], '*')))
            except:
                pass

        # Saving images to their cluster directories
        for fname in self.file_names:
            img_dest_path = os.path.join(cluster_img_dest[self.cluster_result[fname]], fname[fname.rfind('/')+1 : ])
            print("Saving image {} to {}".format(fname, img_dest_path))
            copyfile(fname, img_dest_path)
