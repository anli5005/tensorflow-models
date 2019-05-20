import tensorflow as tf
import os
import xml.etree.ElementTree as ET

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', os.path.join(os.getcwd(), 'object_detection/tf_record_thing'), '')
FLAGS = flags.FLAGS

def create_tf_example(example):
    tree = ET.parse(example)
    annotation = tree.getroot()

    size = annotation.find("size")

    height = int(size.find("height").text) # Image height
    width = int(size.find("width").text) # Image width
    filename = annotation.find("filename").text + ".jpg" # Filename of the size.image. Empty if image is not from file
    encoded_image_data = None # Encoded image bytes
    image_format = "jpeg" # b'jpeg' or b'png'

    obj = annotation.find("object")
    bndbox = obj.find("bndbox")

    xmins = [int(bndbox.find("xmin").text) / width] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [int(bndbox.find("xmax").text) / width] # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [int(bndbox.find("ymin").text) / height] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [int(bndbox.find("ymax").text) / height] # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    annotations_path = os.path.join(os.getcwd(), "annotations")
    paths = (os.path.join(annotations_path, name) for name in os.listdir(annotations_path))
    examples = (os.path.join(path, name) for path in paths for name in os.listdir(path))

    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()