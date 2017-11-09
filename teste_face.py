import os
import cv2
import copy
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image


DIR_ATUAL = os.getcwd()

GRAPH = "/home/cleverson/pessoal/cleverson/workspace/mestrado/tensorflow-face-object-detector-tutorial/model/frozen_inference_graph.pb"

IMAGEM_TESTE = os.path.join(DIR_ATUAL, "pessoas","face_0.png")

PATH_TO_LABELS = os.path.join(DIR_ATUAL, 'object_detection', 
    'data', 'face.pbtxt')

# Rede Neural 
# GRAPH = "/home/cleverson/pessoal/mestrado/redes_neurais/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb"

NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

sess = tf.Session(graph=detection_graph)

img = IMAGEM_TESTE
frame = cv2.imread(img)
image_np = frame[:,:,::-1]

image_np_expanded = np.expand_dims(image_np, axis=0)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
scores = detection_graph.get_tensor_by_name('detection_scores:0')
classes = detection_graph.get_tensor_by_name('detection_classes:0')


(boxes, scores, classes) = sess.run( [boxes, scores, classes], 
    feed_dict={image_tensor: image_np_expanded})


i = 0


vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8)

output_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
cv2.imshow('Video', output_rgb)


cv2.waitKey(0)
cv2.destroyAllWindows()

#print(boxes)