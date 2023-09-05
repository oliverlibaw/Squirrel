import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        for member in root.findall('object'):
            class_name = member.find('name').text
            xmin = int(member.find('bndbox/xmin').text)
            ymin = int(member.find('bndbox/ymin').text)
            xmax = int(member.find('bndbox/xmax').text)
            ymax = int(member.find('bndbox/ymax').text)
            xml_list.append([filename, width, height, class_name, xmin, ymin, xmax, ymax])
    
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    for folder in ['train', 'validation']:
        image_path = os.path.join(os.getcwd(), 'images', folder)
        xml_df = xml_to_csv(image_path)
        output_csv_path = os.path.join(os.getcwd(), 'images', f'{folder}_labels.csv')
        xml_df.to_csv(output_csv_path, index=None)
        print(f'Successfully converted xml to csv for {folder}.')

if __name__ == "__main__":
    main()
