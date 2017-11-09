import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw


from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

# Rede treinada para deteccao de objetos.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', 
    MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 
    'data', 'mscoco_label_map_person.pbtxt')

PATH_TO_LABELS_FACE = os.path.join(CWD_PATH, 'object_detection', 
    'data', 'face.pbtxt')

GRAPH_FACE = "/home/cleverson/pessoal/cleverson/workspace/mestrado/tensorflow-face-object-detector-tutorial/model/frozen_inference_graph.pb"

NUM_CLASSES = 1

NUM_CLASSES_FACE = 1

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

label_map_face = label_map_util.load_labelmap(PATH_TO_LABELS_FACE)
categories_face = label_map_util.convert_label_map_to_categories(
    label_map_face, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index_face = label_map_util.create_category_index(categories_face)


def detectar_humanos(image_np, sess, detection_graph, sess_face, detection_graph_face):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    image_tensor_face = detection_graph_face.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    boxes_face = detection_graph_face.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    scores_face = detection_graph_face.get_tensor_by_name('detection_scores:0')
    classes_face = detection_graph_face.get_tensor_by_name('detection_classes:0')

    image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
    im_width, im_height = image_pil.size
    

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    (boxes_face, scores_face, classes_face) = sess_face.run(
        [boxes_face, scores_face, classes_face],
        feed_dict={image_tensor_face: image_np_expanded})

    i = 0
    for pbox_face in np.squeeze(boxes_face):

        if scores[0][i] < .5:
            continue

        y1 = int(pbox_face[0] * im_height)
        x1 = int(pbox_face[1] * im_width)
        y2 = int(pbox_face[2] * im_height)
        x2 = int(pbox_face[3] * im_width)

        (left, right, top, bottom) = ( x1, x2, y1, y2)

        image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
        draw = ImageDraw.Draw(image_pil)

        draw.line([(left, top), (left, bottom), (right, bottom),
                 (right, top), (left, top)], width=4, fill='red')
        np.copyto(image_np, np.array(image_pil))
        i += 1


    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    detection_graph_face = tf.Graph()
    with detection_graph_face.as_default():
        od_graph_def_face = tf.GraphDef()
        with tf.gfile.GFile(GRAPH_FACE, 'rb') as fid_face:
            serialized_graph_face = fid_face.read()
            od_graph_def_face.ParseFromString(serialized_graph_face)
            tf.import_graph_def(od_graph_def_face, name='')

        sess_face = tf.Session(graph=detection_graph_face)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detectar_humanos(frame_rgb, sess, detection_graph, sess_face, detection_graph_face))

    fps.stop()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()

    while True:  # fps._numFrames < 120
        frame = video_capture.read()
        input_q.put(frame)

        t = time.time()

        output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', output_rgb)
        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
