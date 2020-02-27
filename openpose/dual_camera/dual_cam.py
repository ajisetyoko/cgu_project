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

    video_capture_0 = cv2.VideoCapture(2)
    video_capture_1 = cv2.VideoCapture(0)

    video_capture_0.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    video_capture_0.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    video_capture_0.set(cv2.CAP_PROP_FPS,15)

    video_capture_1.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    video_capture_1.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    video_capture_1.set(cv2.CAP_PROP_FPS,15)

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
            dual_frame  = np.concatenate((frame0, frame1), axis=1)
            image_ori   = dual_frame.copy()
            image       = dual_frame
            humans      = e.inference(image,resize_to_default=True, upsample_size=6.0)
            image_shape = image.shape[:2]
            start_pos   = [pr[1],pr[0]]
            if i<1:
                #inisialize
                point = stabilkanpointd(humans.copy(),image_shape,0,start_pos)
            else:
                #update
                point.update(humans.copy(),i)
                distance = point.dis_cal()
            image = TfPoseEstimator.draw_humans(image_ori,image,start_pos, humans, imgcopy=False)

            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10)
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 4)

            # cv2.rectangle(image,(pr[0],pr[1]),(pr[2],pr[3]),(0,0,0),2) #Rectangle of the ROI

            last_skeleton = point.camera_parsing()
            try:
                print('Camera 1 ',last_skeleton[0][0][0:10])
                print('Camera 2 ',last_skeleton[1][0][0:10])
            except IndexError:
                pass
            try:
                print('Camera 1 ',last_skeleton[0][0][0:10])
            except IndexError:
                pass

            image_1 = image[0:480,0:640]
            image_2 = image[0:480,640:1280]

            cv2.imshow('frame_1', image_1)
            cv2.imshow('frmae_2', image_2)
            out.write(image_1)
            fps_time = time.time()
            i +=1

            # Stopping Method
            if cv2.waitKey(1) == 27:
                saver_json(point,ori_w,ori_h)
                break
        else:
            saver_json(point,ori_w,ori_h)
            break
    cv2.destroyAllWindows()
    out.release()
    cap.release()

logger.debug('finished+')
