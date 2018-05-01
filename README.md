# B657_Final_Project
Our experiments consist of two parts of source code:

<b> Hypothesis I </b> 
<br />
<ol> We used a pre-trained Mask R-CNN to do inference on our dataset. The source code of the network and the model is at https://github.com/facebookresearch/Detectron. The file we mainly used was tools/infer_simple.py with model cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml. This repo is posted by Facebook. </ol>
<ol> Our scripts to do basic statistics on object detection results and to randomly sample images are in directory Mask_RCNN_results. These are done by us. </ol>

<b> Hypothesis II </b>
<ol> We trained beta-VAEs to test out our second hypothesis. Our source code for the network is based on https://github.com/chaitanya100100/VAE-for-Image-Generation.</ol>
<ol> In directory homeview_src, these files are based on the above repo: homeview_generate.py, homeview_params.py, homeview_params_cluster.py, homeview_train.py, homeview_cluster.py with minor modifications to change the network from a VAE to a beta-VAE and to work on our dataset. </ol>
<ol> In directory homeview_src, data_manager.py is based on on Tensorflow's tutporial at https://www.tensorflow.org/programmers_guide/datasets#decoding_image_data_and_resizing_it </ol>
<ol> In the same directory, cluster.py is our own work based on scikit-learn tutorial http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html </ol>
<ol> Model files are in the models directory. Files with homeview_ld_68_conv_5_id_1060_e_1000 in their names are model for 68 dimensions in the latent layer and files with homeview_ld_168_conv_4_id_1360_e_500 in their names are model for 168 dimensions in the latent layer. </ol>

<br />

<b> Dataset </b>
<br />
Unfortunately our dataset is under IRB restriction so we can't share them here.
