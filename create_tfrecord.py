import os
import glob
import pandas as pd
import xml.etree.etree as ET
import tensorflow as tf
from object_detection.utils import dataset_util 

train_path = '/content/drive/MyDrive/squirrels/train'
test_path = '/content/drive/MyDrive/squirrels/test'

label_map = {1: 'Squirrel'} 

def xml_to_csv(path):
  xml_list = []
  for xml_file in glob.glob(path + '/*.xml'):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
      value = (root.find('filename').text,  
               int(root.find('size')[0].text),
               int(root.find('size')[1].text),
               label_map[int(member[0].text)],
               int(member[4][0].text),
               int(member[4][1].text),
               int(member[4][2].text),
               int(member[4][3].text)
              )
      xml_list.append(value)
  column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
  xml_df = pd.DataFrame(xml_list, columns=column_name)
  return xml_df

def create_tf_example(group, path):
  with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)

  width, height = image.size

  filename = group.filename.encode('utf8')
  image_format = b'jpg'
  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  classes_text = []
  classes = []

  for index, row in group.object.iterrows():
    xmins.append(row['xmin'] / width)
    xmaxs.append(row['xmax'] / width)
    ymins.append(row['ymin'] / height)
    ymaxs.append(row['ymax'] / height)
    classes_text.append(row['class'].encode('utf8'))
    classes.append(class_text_to_int(row['class']))
  
  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(filename),
    'image/source_id': dataset_util.bytes_feature(filename),
    'image/encoded': dataset_util.bytes_feature(encoded_jpg),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def class_text_to_int(row_label):
  return list(label_map.keys())[list(label_map.values()).index(row_label)]  

def split(df, group):
  data = namedtuple('data', ['filename', 'object'])
  gb = df.groupby(group)
  return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def main():
  train_xml_df = xml_to_csv(train_path) 
  test_xml_df = xml_to_csv(test_path)

  # Create TFRecord files
  train_writer = tf.python_io.TFRecordWriter('train.record')
  test_writer = tf.python_io.TFRecordWriter('test.record')

  train_grouped = split(train_xml_df, 'filename')
  for group in train_grouped:
    tf_example = create_tf_example(group, train_path)
    train_writer.write(tf_example.SerializeToString())

  test_grouped = split(test_xml_df, 'filename')
  for group in test_grouped: 
    tf_example = create_tf_example(group, test_path)
    test_writer.write(tf_example.SerializeToString())

  train_writer.close()
  test_writer.close()

  print('Successfully created TFRecord files.')
  
main()
