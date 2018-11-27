

import cv2
import os
import copy
import json
from os.path import expanduser
import argparse
import datetime
import numpy as np
import pickle
import copy

class RotatedRect:
	def __init__(self):
		self.cx = 0
		self.cy = 0
		self.width = 0
		self.height = 0
		self.angle = 0
		self.label = 0

def read_grasping_rectangle_file(file_path,rect_type):

	with open(file_path,"r") as f:
		lines= f.readlines()

	rectangle_count = int(len(lines)/4)

	rectangles = np.zeros((rectangle_count,4,1,2)).astype(np.int32)

	rotated_rects = []

	rect = np.zeros((4,1,2))
    #y = []
	count = 0
	for i,line in enumerate(lines):
		rect[count,0,0] = round(float(line.split("\r\n")[0].split(" ")[0]))
		rect[count,0,1] = round(float(line.split("\r\n")[0].split(" ")[1]))

		i = i+1
		count += 1
		if( i % 4 == 0 ):
			count = 0

			#print rect

			rectangles[int(i/4)-1] =  rect
            #print x
            #x = x.reshape((-1,1,2))
            #print x
			rect = np.zeros((4,1,2))
	print len(rectangles)

	rotated_rect = RotatedRect()

	for rect in rectangles:
		angle = 0

		edge1 = np.sqrt((rect[0,0,0] - rect[1,0,0]) * (rect[0,0,0] - rect[1,0,0]) + (rect[0,0,1] - rect[1,0,1]) * (rect[0,0,1] - rect[1,0,1]))
		edge2 = np.sqrt((rect[1,0,0] - rect[2,0,0]) * (rect[1,0,0] - rect[2,0,0]) + (rect[1,0,1] - rect[2,0,1]) * (rect[1,0,1] - rect[2,0,1]))

		if edge1 < 5.0 or edge2 < 5.0:
			continue

		if edge1 > edge2:
			rotated_rect.width = edge1

			rotated_rect.height = edge2

			if rect[0,0,0] - rect[1,0,0] != 0:
				angle = -np.arctan(float(rect[0,0,1]- rect[1,0,1]) / float(rect[0,0,0] - rect[1,0,0])) / 3.1415926 * 180
			else:
				angle = 90.0
		else:
			rotated_rect.width = edge2

			rotated_rect.height = edge1

			if rect[1,0,0] - rect[2,0,0] != 0:
				angle = -np.arctan(float(rect[1,0,1]- rect[2,0,1]) / float(rect[1,0,0] - rect[2,0,0])) / 3.1415926 * 180
			else:
				angle = 90.0

		if angle < -45.0:
			angle = angle + 180

		rotated_rect.angle = angle

		rotated_rect.cx = (rect[0,0,0]+rect[2,0,0])/2

		rotated_rect.cy = (rect[0,0,1]+rect[2,0,1])/2

		rotated_rect.label = rect_type

		rotated_rects.append(copy.copy(rotated_rect))
        #print rotated_rects
		#rotated_rect.reset()

	#print "rotated_rects",rotated_rects[0],rotated_rects[1]
	return rotated_rects






def get_Cornell_grasping_rect(mode, task, prefetched):
	#DATASET_DIR = '/home/hakan/cornell_grasping_dataset_wrapper/shuffle0/'
	#DATASET_TRAINING_DIR = '/home/hakan/cornell_grasping_dataset_wrapper/shuffle0/training'

	#DATASET_DIR = '/home/hakan/trial2/shuffle0/'
	#DATASET_TRAINING_DIR = '/home/hakan/trial2/shuffle0/training'

	#DATASET_DIR = '/home/hakan/shuffle0training/'
	#DATASET_TRAINING_DIR = '/home/hakan/shuffle0training/training'

	#DATASET_DIR = '/home/hakan/shuffle0training2/'
	#DATASET_TRAINING_DIR = '/home/hakan/shuffle0training2/training'

	#DATASET_DIR = '/home/hakan/shuffle1/'
	#DATASET_TRAINING_DIR = '/home/hakan/shuffle1/training'

	DATASET_DIR = '/home/hakan/shuffle2/augmentedtraining'
	DATASET_TRAINING_DIR = '/home/hakan/shuffle2/augmentedtraining/training'

	#DATASET_DIR = '/home/hakan/shuffle2/'
	#DATASET_TRAINING_DIR = '/home/hakan/shuffle2/training'


	im_infos = []
	data_list = []
	gt_list = []
	img_type = ['png']
	cls_list = {'background':0, 'negative':1, 'positive':2}
	#cls_list = {'background':0, 'positive':1}
	if not prefetched:
		# training set contains 7200 images with
		image_file_paths = []
		image_names = []
		if mode == "train":
			for i,file in enumerate(os.listdir(DATASET_TRAINING_DIR)):
				if file.endswith(".png"):
					path = os.path.join(DATASET_TRAINING_DIR,file)
					image_file_paths.append(path)
					image_names.append(file)

				#if i == 46:
				#	break
			print len(image_file_paths)
			#return

			for i,image_file in enumerate(image_file_paths):
				#print i

				image_name = image_names[i]

				base_file_name = image_name[:-5]

				negative_rect_file_name = base_file_name + "cneg.txt"

				positive_rect_file_name = base_file_name + "cpos.txt"
				negative_rect_file_path = os.path.join(DATASET_TRAINING_DIR,negative_rect_file_name)
				positive_rect_file_path = os.path.join(DATASET_TRAINING_DIR,positive_rect_file_name)
				#print positive_rect_file_name
				#print image_file
				img = cv2.imread(image_file)
				boxes = []
				#rects = read_grasping_rectangle_file(negative_rect_file_path,"negative")
				#for rect in rects:
				#	boxes.append([rect.cx, rect.cy, rect.height, rect.width, rect.angle, rect.label])
				rects = read_grasping_rectangle_file(positive_rect_file_path,"positive")
				for rect in rects:
					#print rect.cx
					#print rect.cy
					boxes.append([rect.cx, rect.cy, rect.height, rect.width, rect.angle, rect.label])

				cls_num = 2
				if task == "multi_class":
					cls_num = len(cls_list.keys())

				len_of_bboxes = len(boxes)
				#print "bboxes ", len_of_bboxes
				gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
				gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
				overlaps = np.zeros((len_of_bboxes, cls_num), dtype=np.float32)
				seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)
				#print boxes
				if task == "multi_class":
					gt_boxes = [] #np.zeros((len_of_bboxes, 5), dtype=np.int16)
					gt_classes = [] #np.zeros((len_of_bboxes), dtype=np.int32)
					overlaps = [] #np.zeros((len_of_bboxes, cls_num), dtype=np.float32) #text or non-text
					seg_areas = [] #np.zeros((len_of_bboxes), dtype=np.float32)
				for idx in range(len(boxes)):
					#print boxes[idx]
					if task == "multi_class":
						if not boxes[idx][5] in cls_list:
							print (boxes[idx][5] + " not in list")
							continue
						gt_classes.append(cls_list[boxes[idx][5]]) # cls_text
						overlap = np.zeros((cls_num))
						overlap[cls_list[boxes[idx][5]]] = 1.0 # prob
						overlaps.append(overlap)
						seg_areas.append((boxes[idx][2]) * (boxes[idx][3]))
						gt_boxes.append([boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]])
					else:
						gt_classes[idx] = 1 # cls_text
						overlaps[idx, 1] = 1.0 # prob
						seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])
						gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]

				if task == "multi_class":
					gt_classes = np.array(gt_classes)
					overlaps = np.array(overlaps)
					seg_areas = np.array(seg_areas)
					gt_boxes = np.array(gt_boxes)

				#print ("boxes_size:", gt_boxes.shape[0])

				if gt_boxes.shape[0] > 0:
					max_overlaps = overlaps.max(axis=1)
					# gt class that had the max overlap
					max_classes = overlaps.argmax(axis=1)

					im_info = {
						'gt_classes': gt_classes,
						'max_classes': max_classes,
						'image': image_file,
						'boxes': gt_boxes,
						'flipped' : False,
						'gt_overlaps' : overlaps,
						'seg_areas' : seg_areas,
						'height': img.shape[0],
						'width': img.shape[1],
						'max_overlaps' : max_overlaps,
						'rotated': True
                	}
				#print im_info["gt_classes"]
					im_infos.append(im_info)

		print len(im_infos)
		f_save_pkl = open('Cornell_training_cache.pkl', 'wb')
		pickle.dump(im_infos, f_save_pkl)
		f_save_pkl.close()
		print ("Save pickle done.")
		return im_infos
	else:
		if mode == "train":
			f_pkl = open('Cornell_training_cache.pkl', 'rb')
			im_infos = pickle.load(f_pkl)
		if mode == "validation":
			f_pkl = open('ICDAR2017_validation_cache.pkl', 'rb')
			im_infos = pickle.load(f_pkl)
	return im_infos
