from __future__ import print_function

import cv2
import numpy as np
import matplotlib.patches as patches
import Image
from matplotlib import pyplot as plt
import imageio
import os
import glob
import time
import sys


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def find_face(inp_img):
    gray_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 5)
    if len(faces) != 1:
        raise AssertionError("Different than one face found: found {}!".format(len(faces)))
    else:
        return faces[0]

def extract_face(inp_img, final_size=(48,48), prev_pos=None):
    try:
        x,y,w,h = find_face(inp_img)
    except AssertionError as e:
        if prev_pos == None:
            raise AssertionError('!')
        else:
            x,y,w,h = prev_pos
    im = Image.fromarray(inp_img[y:y+w,x:x+h])
    im.thumbnail(final_size, Image.ANTIALIAS)
    extracted_face = np.array(im.getdata(),np.uint8).reshape(
        im.size[1], im.size[0], 3)
    return extracted_face, (x,y,w,h)

print('Functions declared.')

GRID_DIR = '/home/dneil/datasets/grid/video/'
output_folder = GRID_DIR+'np_data/'
final_size = (48,48)

all_people = [x[0] for x in os.walk(GRID_DIR)][1:]
for person_dir in all_people:
    start_time = time.time()
    person_key = person_dir.split('/')[-1]
    all_files_for_person = glob.glob(person_dir+'/*.mpg')
    for filename in all_files_for_person:        
        file_key = filename.split('/')[-1][:-4]
        final_key = '{}-{}'.format(person_key, file_key)
        try:
            old_pos = None
            # Load the data
            vid = imageio.get_reader(filename, 'ffmpeg')
            #   Prebuild the data store
            out_data = np.zeros( (vid.get_length(), final_size[1], final_size[0], 3) ).astype('uint8')
            #   Get all the frames
            for i in range(vid.get_length()):
                #   Extract the face
                try:
                    out_data[i], old_pos = extract_face(vid.get_data(i), prev_pos=old_pos)
                except Exception as e:
                    print('An inner error occurred: ', e, filename, i)                    
            # Save the result
            np.save(output_folder+final_key+'_vid.npy', out_data)
        except Exception as e:
            print('An error occurred: ', e)
        print('.', end="")
        sys.stdout.flush()
    print("\nCompleted {} in {} seconds.".format(person_key, time.time()-start_time))
print('Done.')