# @Author: ajisetyoko <simslab-cs>
# @Date:   2020-02-25T17:31:20+08:00
# @Email:  aji.setyoko.second@gmail.com
# @Last modified by:   simslab-cs
# @Last modified time: 2020-02-27T18:35:41+08:00



import os
os.chdir('../../tf-pose-estimation/')
import sys
sys.path.append('/home/simslab-cs/Documents/lab-project/repository/tf-pose-estimation/')
print(os.getcwd())
import argparse;import logging;import time

import cv2;import numpy as np;import tensorflow as tf
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from stabilize import stabilkanpointd

import json;import numpy as np

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

from selfmake_helper import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368',
                        help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--resize-out-ratio', type=float, default=2.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()

    #GPU-CPU Setter
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    #Model Init
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w,h = int(432/2),int(368/2)# w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h),
                        tf_config=config,manual_roi=(474,551))

    video_capture_1 = cv2.VideoCapture('dual_camera/output1.mp4')
    video_capture_0 = cv2.VideoCapture('dual_camera/output2.mp4')

    cap = video_capture_0

    #Video Writer Parameter
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    ori_w  = 640*2 ; ori_h  = 480 ; channel= 3
    background = np.zeros((ori_h,ori_w,channel))
    output_fps = 25

    out = cv2.VideoWriter('labort_test/output.avi',fourcc, output_fps, (int(cap.get(3)), int(cap.get(4))))

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    i = 0
    pr = [0,0,0,0]
    while cap.isOpened():
        ret0, frame0 = video_capture_0.read()
        ret1, frame1 = video_capture_1.read()
        if (ret0 and ret1):
            image1, image2 = frame0.copy(),frame1.copy()
            humans1      = e.inference(image1,resize_to_default=True, upsample_size=6.0)
            humans2      = e.inference(image2,resize_to_default=True, upsample_size=6.0)
            image_shape  = image1.shape[:2]
            start_pos    = [pr[1],pr[0]]
            if i<1:
                #inisialize
                point1 = stabilkanpointd(humans1.copy(),image_shape,0,start_pos)
                point2 = stabilkanpointd(humans2.copy(),image_shape,0,start_pos)
            else:
                #update
                point1.update(humans1.copy(),i)
                point2.update(humans2.copy(),i)
                distance = point1.dis_cal()
            bx1 = np.zeros((480,640,3))
            bx  = np.zeros((480,640,3))
            image1 = TfPoseEstimator.draw_humans(bx1,image1,start_pos, humans1, imgcopy=False)
            image2 = TfPoseEstimator.draw_humans(bx,image2,start_pos, humans2, imgcopy=False)
            cv2.putText(image1, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10)
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 4)

            dual_frame_in  = np.concatenate((frame0, frame1), axis=1)
            dual_frame_out = np.concatenate((image1,image2), axis =1)
            c = cv2.resize(dual_frame_in,(int(1.5*640),int(1.5*240)))
            cv2.imshow('frame_1', c)
            c = cv2.resize(dual_frame_out,(int(1.5*640),int(1.5*240)))
            cv2.imshow('frmae_2', c)
            fps_time = time.time()
            i +=1

            # Stopping Method
            if cv2.waitKey(1) == 27:
                saver_json(point1,ori_w,ori_h)
                break
        else:
            saver_json(point1,ori_w,ori_h)
            break
    cv2.destroyAllWindows()
    out.release()
    cap.release()

logger.debug('finished+')
