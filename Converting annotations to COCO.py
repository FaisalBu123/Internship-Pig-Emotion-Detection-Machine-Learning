import os
from xml.dom import minidom

out_dir = "C:\\Users\\Faisal\\Downloads\\Internship summer 2024\\August new images w updates from disi\\Data 9\\Data\\labels\\train"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

file_path = "C:\\Users\\Faisal\\Downloads\\Internship summer 2024\\August new images w updates from disi\\task_larger dataset pigs annotation_annotations_2024_09_10_12_30_39_cvat for images 1.1\\annotations.xml"
file = minidom.parse(file_path)

images = file.getElementsByTagName('image')

for image in images:
    width = int(image.getAttribute('width'))
    height = int(image.getAttribute('height'))
    name = image.getAttribute('name')

    # Get all bounding boxes and keypoints for pigs
    bboxes = image.getElementsByTagName('box')
    keypoints = image.getElementsByTagName('points')

    # Open label file for writing
    label_file = open(os.path.join(out_dir, name[:-4] + '.txt'), 'w')

    # Loop through each bounding box and its corresponding keypoints
    for i, bbox in enumerate(bboxes):
        xtl = int(float(bbox.getAttribute('xtl')))
        ytl = int(float(bbox.getAttribute('ytl')))
        xbr = int(float(bbox.getAttribute('xbr')))
        ybr = int(float(bbox.getAttribute('ybr')))
        w = xbr - xtl
        h = ybr - ytl

        # Ensure there is a corresponding keypoints set for this box
        if i < len(keypoints):
            kp = keypoints[i]

            # Write the bounding box and center coordinates
            label_file.write('0 {} {} {} {} '.format(
                str((xtl + (w / 2)) / width),  # center x
                str((ytl + (h / 2)) / height),  # center y
                str(w / width),  # width normalized
                str(h / height)  # height normalized
            ))

            # Get keypoints for this specific bounding box
            points = kp.attributes['points'].value.split(';')
            points_ = []
            for p in points:
                p1, p2 = p.split(',')
                points_.append([int(float(p1)), int(float(p2))])

            # Write keypoints normalized by image dimensions
            for p_, point in enumerate(points_):
                label_file.write('{} {}'.format(point[0] / width, point[1] / height))
                if p_ < len(points_) - 1:
                    label_file.write(' ')
                else:
                    label_file.write('\n')

    # Close the label file after writing all bounding boxes and keypoints
    label_file.close()