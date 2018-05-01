# Basic statistics and plots on object detection results from Mask R-CNN

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Utils..
# Hash of super class : object categories
'''
12 super classes from COCO
object_classes_super = { 'person' : ['person'],
                         'vehicle' : ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
                         'outdoor' : ['traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench'],
                         'animal' : ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
                         'accessory' : ['hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase'],
                         'sports' : ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
                         'kitchen' : ['bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'],
                         'food' : ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
                         'furniture' : ['chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door'],
                         'electronic' : ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'],
                         'appliance' : ['microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender'],
                         'indoor' : ['book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush'] }
'''

'''
# Person vs. Non-person
object_classes_super = { 'person' : ['person'], 'non-person' : ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush'] }
'''

'''
# Person vs. Indoor vs. Outdoor
object_classes_super = { 'person' : ['person']
                         'indoor' : [    ] }
'''

# Classes based on Communicative Development Inventory Words and Sentences
object_classes_super = { 'person' : [ 'person' ],
                         'animal' : [ 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe' ],
                         'food and drink' : [ 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake' ],
                         'furniture and rooms': [ 'suitcase', 'bench', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door','tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors' ],
                         'toys' : [ 'teddy bear'],
                         'clothing' : [ 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie' ],
                         'vehicles' : [ 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
                         'small household items' : [ 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'hair drier', 'toothbrush', 'hair brush'],
                         'outside things' : ['traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket' ]
}

# Make bar graph based on given data, assuming input are arrays of arrays of x and y axis info
def bar_graph(x_axis, y_axis, labels=[], file_name='res.png'):
    # create plot
    #fig, ax = plt.subplots()
    fig = plt.figure()
    index = np.arange(len(x_axis))
    bar_width = 0.1
    opacity = 0.8
    
    if len(labels) == 0:
        rects = plt.bar(index, y_axis, bar_width, alpha=opacity, color='b')
    else:
        colors = ['deepskyblue', 'aqua', 'r', 'c', 'navy', 'g', 'cyan', 'b', 'yellow', 'lightseagreen', 'violet', 'darkblue']
        rects = [ plt.bar(index + bar_width*i, y_axis[i], bar_width, alpha=opacity, color=colors[i], label=labels[i]) for i in range(len(y_axis)) ]

    plt.title('Avg Number of objects by number of images')
    plt.xlabel('Subjects')
    plt.ylabel('Number of objects')
    plt.xticks(index + bar_width, x_axis)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

# Get total # of images for each child based on age (os_type = 0)
def get_num_of_imgs(img_location, num_of_img_per_child):
    for file_name in os.listdir(img_location):
        if file_name.endswith('.jpg'):
            subject_name = file_name[:file_name.find('_')]
            if subject_name not in num_of_img_per_child.keys():
                num_of_img_per_child[subject_name] = 1
            else:
                num_of_img_per_child[subject_name] += 1

# Get the number of objects per child  (op_type = 1)
def get_avg_objs_per_child(child_img_num, class_res, num_of_objects_per_child, num_of_images_per_child):
    with open(child_img_num) as f:
        for l in f:
            subject = l.strip()[0 : l.find(':')]
            num_of_imgs = l.strip()[l.find(' ')+1:]
            num_of_images_per_child[subject] = float(num_of_imgs)
    f.close()
    # Find out number of objects by counting the number of confidence scores
    with open(class_res) as f:
        for l in f:
            num_of_objects = len([ float(s) for s in re.findall(r'-?\d+\.?\d*', l.strip()[l.find(' ')+1:]) ])
            subject = l.strip()[ :l.find('_')]
            if subject not in num_of_objects_per_child.keys():
                num_of_objects_per_child[subject] = num_of_objects
            else:
                num_of_objects_per_child[subject] += num_of_objects
    f.close()

    # Get average based on number of images
    for k in num_of_objects_per_child.keys():
        if k in num_of_images_per_child.keys():
            num_of_objects_per_child[k] /= num_of_images_per_child[k]

    # Sort the dictionary by keys and plot the result
    num_of_objects_per_child = sorted(num_of_objects_per_child.items())
    x_axis = [ res[0] for res in num_of_objects_per_child ]
    y_axis = [ res[1] for res in num_of_objects_per_child ]
    #print x_axis, y_axis
    bar_graph(x_axis, y_axis)

# Get the number of objects per child divided into object super classes (op_type = 3)
def get_avg_objs_per_child_with_classes(child_img_num, class_res, num_of_objects_per_child, num_of_images_per_child):
    # Get number of images for each child
    with open(child_img_num) as f:
        for l in f:
            subject = l.strip()[0 : l.find(':')]
            num_of_imgs = l.strip()[l.find(' ')+1:]
            num_of_images_per_child[subject] = float(num_of_imgs)
    f.close()
    
    # Save object information for eac subject
    sub_object_info = {}
    with open(class_res) as f:
        for l in f:
            subject_id = l.split()[0][0 : l.split()[0].find('_')]
            if subject_id not in sub_object_info.keys():
                sub_object_info[subject_id] = l.strip()[l.find(' ')+1:].split()
            else:
                sub_object_info[subject_id] += l.strip()[l.find(' ')+1:].split()
    f.close()

    # Get the names of super classes
    object_classes = [ key for key in object_classes_super.keys() ]
    # Reverse keys and values in object_classes_super                                                                                                                            
    object_classes_super_inv = {}
    for k in object_classes_super.keys():
        for object_class in object_classes_super[k]:
            object_classes_super_inv[object_class] = k
            
    # Build a hash that saves subject name and object count for each super class { subj : {obj_class : count} }
    for k in sub_object_info.keys():
        num_of_objects_per_child[k] = { obj_class : 0 for obj_class in object_classes }
    # Extract object information for each child
    for subj in sub_object_info.keys():
        objects = sub_object_info[subj]
        n = 0
        while n < len(objects):
            object_class = objects[n]
            if object_class in object_classes_super_inv.keys():
                num_of_objects_per_child[subj][object_classes_super_inv[object_class]] += 1
                n += 1
            elif object_class not in re.findall(r'-?\d+\.?\d*',object_class):
                num_of_objects_per_child[subj][object_classes_super_inv[ object_class + ' ' + objects[n+1]] ] += 1
                n += 2
            else:
                n += 1
    #print num_of_objects_per_child
    #print '# of images per kid: ', num_of_images_per_child
    
    # Get average based on number of images
    for k in num_of_objects_per_child.keys():
        for obj in num_of_objects_per_child[k]:
            num_of_objects_per_child[k][obj] /= num_of_images_per_child[k]
    print num_of_objects_per_child

    # Sort the dictionary by keys and plot the result
    num_of_objects_per_child = sorted(num_of_objects_per_child.items())
    x_axis = [ res[0] for res in num_of_objects_per_child ]
    y_axis = np.array([ res[1].values() for res in num_of_objects_per_child ]).transpose()
    labels = [ res[1].keys() for res in num_of_objects_per_child][0]
    #print 'x axis: ', x_axis
    #print 'y axis: ', y_axis
    #print 'y labels: ', labels
    bar_graph(x_axis, y_axis, labels)

# Get the number of objects per age group divided into super classes  (op_type = 5)
# parameters: child_num_image.txt, array of class_res_(3,8,12), array of num_of_objects and num_of_images_per_child
def get_avg_objs_age_classes(child_img_num, class_res, num_of_objects, num_of_images_per_child):
    # Get number of images for each child
    with open(child_img_num) as f:
        for l in f:
            subject = l.strip()[0 : l.find(':')]
            num_of_imgs = l.strip()[l.find(' ')+1:]
            num_of_images_per_child[subject] = float(num_of_imgs)
    f.close()
    
    # Extract object information for each age group [ 1-3, 6-8, 11-12 ]
    object_info = [ [], [], [] ]
    # Get subject ID in each age group
    subject_info = [ [], [], [] ]
    for i in range(len(class_res)):
        with open(class_res[i]) as f:
            for l in f:
                object_info[i].append(l.strip()[l.find(' ')+1:].split())
                if l.strip()[0:l.find('_')] not in subject_info[i]:
                    subject_info[i].append(l.strip()[0:l.find('_')])
    #print subject_info

    object_classes = [ obj_class for key in object_classes_super.keys() for obj_class in  object_classes_super[key] ]
    # Reverse keys and values in object_classes_super
    object_classes_super_inv = {}
    for k in object_classes_super.keys():
        for object_class in object_classes_super[k]:
            object_classes_super_inv[object_class] = k
    
    # Update number of objects in each super class
    for i in range(len(object_info)):
        for objects in object_info[i]:
            n = 0
            while n < len(objects):
                object_class = objects[n]
                if object_class in object_classes:
                    num_of_objects[i][object_classes_super_inv[object_class]] += 1
                    n += 1
                elif object_class not in re.findall(r'-?\d+\.?\d*',object_class):
                    num_of_objects[i][object_classes_super_inv[ object_class + ' ' + objects[n+1]] ] += 1
                    n += 2
                else:
                    n += 1
    
    # Plot bar graph
    x_axis = ['01_to_03_months', '06_to_08_months', '11_to_12_months']
    y_axis = np.array([ obj_info.values() for obj_info in num_of_objects ], np.float).transpose()
    # Average by number of subjects in each age group
    #for i in range(y_axis.shape[1]):
    #print len(subject_info[i]), y_axis[:,i]
    #y_axis[:,i] /= len(subject_info[i])
    
    # Average by number of images in each age group
    num_of_images_age = []
    for subject in subject_info:
        num_of_images = 0
        for s in subject:
            num_of_images += num_of_images_per_child[s]
        num_of_images_age.append(num_of_images)
    for i in range(y_axis.shape[1]):
        y_axis[:,i] /= num_of_images_age[i]
    #print num_of_images_age
    
    labels = num_of_objects[0].keys()
    print x_axis, y_axis, labels
    bar_graph(x_axis, y_axis, labels)
    #print num_of_objects

                    
################## Main function ##################
# img_location - directory where source images are in
# op_type - what kind of statistics we want to evaluate 
#    0: get number of images per child. This is used to make child_num_image.txt, which is later used to get the number of images per child.
#    1: number of objects per child, averaged over # of images
#    2: number of objects per child, averaged over # of children in each age group
#    3: number of objects per child, averaged over # of images, divided into super classes
#    4: number of objects age, averaged over # of children in each age group, divided into super classes
# Command line inputs
op_type = int(sys.argv[1])
# Stats based on op_type
if op_type == 0:
    (img_location1, img_location2) = sys.argv[2:]
    num_of_img_per_child = {}
    get_num_of_imgs(img_location1, num_of_img_per_child)
    get_num_of_imgs(img_location2, num_of_img_per_child)

elif op_type == 1:
    print 'number of objects per child, averaged over # of images..'
    (child_img_num, class_res) = sys.argv[2:]
    num_of_objects_per_child = {}
    num_of_images_per_child = {}
    get_avg_objs_per_child(child_img_num, class_res, num_of_objects_per_child, num_of_images_per_child)

elif op_type == 2:
    (child_img_num, class_res1, class_res2, class_res3) = sys.argv[2:]
    class_res = [ class_res1, class_res2, class_res3 ]

elif op_type == 3:
    print 'Getting the number of objects in each subclass per kid in each age group...'
    (child_img_num, class_res) = sys.argv[2:]
    num_of_objects_per_child = {}
    num_of_images_per_child = {}
    get_avg_objs_per_child_with_classes(child_img_num, class_res, num_of_objects_per_child, num_of_images_per_child)

elif op_type == 4:
    print 'Get the number of objects per age group divided into super classes..'
    (child_img_num, class_res1, class_res2, class_res3) = sys.argv[2:]
    class_res = [ class_res1, class_res2, class_res3 ]
    num_of_objects = [ { key : 0 for key in object_classes_super.keys() } for i in range(3) ]
    num_of_images_per_child = {}
    get_avg_objs_age_classes(child_img_num, class_res, num_of_objects, num_of_images_per_child)
    
