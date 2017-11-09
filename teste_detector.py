import os
import cv2
import random
import copy
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image


DIR_ATUAL = os.getcwd()

GRAPH = os.path.join(DIR_ATUAL, "object_detection",
    "ssd_mobilenet_v1_coco_11_06_2017","frozen_inference_graph.pb")

IMAGEM_TESTE = os.path.join(DIR_ATUAL, "pessoas","13.jpg")

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(DIR_ATUAL, 'object_detection', 
    'data', 'mscoco_label_map.pbtxt')

PATH_TO_LABELS_FACE = os.path.join(DIR_ATUAL, 'object_detection', 
    'data', 'face.pbtxt')

# Rede Neural 
# GRAPH = "/home/cleverson/pessoal/mestrado/redes_neurais/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb"

GRAPH_FACE = "/home/cleverson/pessoal/cleverson/workspace/mestrado/tensorflow-face-object-detector-tutorial/model/frozen_inference_graph.pb"


NUM_CLASSES = 90

NUM_CLASSES_FACE = 1


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

label_map_face = label_map_util.load_labelmap(PATH_TO_LABELS_FACE)
categories_face = label_map_util.convert_label_map_to_categories(
    label_map_face, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index_face = label_map_util.create_category_index(categories_face)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

detection_graph_face = tf.Graph()
with detection_graph_face.as_default():
    od_graph_def_face = tf.GraphDef()
    with tf.gfile.GFile(GRAPH_FACE, 'rb') as fid_face:
        serialized_graph_face = fid_face.read()
        od_graph_def_face.ParseFromString(serialized_graph_face)
        tf.import_graph_def(od_graph_def_face, name='')

sess = tf.Session(graph=detection_graph)
sess_face = tf.Session(graph=detection_graph_face)

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

image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
im_width, im_height = image_pil.size

print(im_width)
print(im_height)

i = 0
for pbox in np.squeeze(boxes):

    if scores[0][i] < 0.5:
        continue

    print("---")
    print(scores[0][i])
    print(pbox)


    y1 = int(pbox[0] * im_height)
    x1 = int(pbox[1] * im_width)
    y2 = int(pbox[2] * im_height)
    x2 = int(pbox[3] * im_width)

    print(y1, x1, y2, x2)

    frame_face = copy.copy(frame[y1:y2, x1:x2])
    image_np_face = frame_face[:,:,::-1]

    cv2.imwrite('face_%d.png' % i,frame_face)

    image_np_expanded_face = np.expand_dims(image_np_face, axis=0)

    image_tensor_face = detection_graph_face.get_tensor_by_name('image_tensor:0')
    boxes_face = detection_graph_face.get_tensor_by_name('detection_boxes:0')
    scores_face = detection_graph_face.get_tensor_by_name('detection_scores:0')
    classes_face = detection_graph_face.get_tensor_by_name('detection_classes:0')

    (boxes_face, scores_face, classes_face) = sess_face.run( 
        [boxes_face, scores_face, classes_face], 
        feed_dict={image_tensor_face: image_np_expanded_face})

    im_width_face, im_height_face, d = frame_face.shape

    print("Numero de faces %d " % len(boxes_face))
    j = 0
    for pbox_face in np.squeeze(boxes_face):

        print("Score %f " % scores_face[0][j])
        if scores_face[0][j] < 0.5:
            continue

        y1f = int(pbox_face[0] * im_height_face)
        x1f = int(pbox_face[1] * im_width_face)
        y2f = int(pbox_face[2] * im_height_face)
        x2f = int(pbox_face[3] * im_width_face)

        (left, right, top, bottom) = ( x1+x1f, x2+x2f, y1+y1f, y2+y2f)

        image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
        draw = ImageDraw.Draw(image_pil)

        draw.line([(left, top), (left, bottom), (right, bottom),
                 (right, top), (left, top)], width=4, fill='red')
        np.copyto(image_np, np.array(image_pil))

        j += 1
    # output_rgb2 = cv2.cvtColor(image_np_person, cv2.COLOR_RGB2BGR)
    # cv2.imshow('Video', output_rgb2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()
    i += 1



print(i)


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