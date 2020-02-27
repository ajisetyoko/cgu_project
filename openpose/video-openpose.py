# @Author: ajisetyoko <simslab-cs>
# @Date:   2020-02-25T17:31:20+08:00
# @Email:  aji.setyoko.second@gmail.com
# @Last modified by:   simslab-cs
# @Last modified time: 2020-02-26T20:52:10+08:00



import argparse;import logging;import time

import cv2;import numpy as np;import tensorflow as tf
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from stabilize import stabilkanpointd,stabilkan

import json;import numpy as np

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def perframe_process(list_bodyparts,frame_number):
    list_human = {}
    list_human['Frame Number'] = frame_number

    for human_number in range(len(list_bodyparts)):
        list_human['Human Confidence Score'] = list_bodyparts[human_number].score
        list_human['Detection no '+str(human_number)] = bodypart_tolist(list_bodyparts[human_number])
    return list_human
def bodypart_tolist(bodyparts):
    y = []
    for part in bodyparts.body_parts.keys():
        parts = ({
            'name'    : str(bodyparts.body_parts[part].get_part_name()),
            'id_part' : part,
            'position': (bodyparts.body_parts[part].x,bodyparts.body_parts[part].y),
            'score'   : bodyparts.body_parts[part].score
        })
        y.append(parts)
    return y

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background
def saver_json(point,ori_w,ori_h,file_name):
    all_data = {}
    all_data['## description'] = ({
        'real_point' : '(int(body_part.x * image_w + image_width+ 0.5),int(body_part.y * image_height +b+ 0.5))',
        'original_width' :ori_w,
        'original_height':ori_h,
        'image_width' : point.w,
        'image_height': point.h,
        'n_frame'     : point.index[-1][0]})
    for number in point.index:
        frame_number = number[0]
        tot_detection= number[1]

        #Saver each frame
        all_data[str(frame_number)] = perframe_process(point.list_human[frame_number],frame_number)

    json_file_name = file_name.split('/')[-1].split('.')[0] + '.json'
    with open(json_file_name, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    # np.save('points.npy',point) for debuging
    print('Detection saved in JSON Format')
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368',
                        help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=False, help='False to show skeleton only.')
    parser.add_argument('--resize-out-ratio', type=float, default=2.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--roi_cut', type=str,default='unset',
                        help = '0,100,0,200 to detect skeleton from a video frame in rectange area with coordinate 0,0 -> 100,200')
    args = parser.parse_args()

    #GPU-CPU Setter
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    #Model Init
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w,h = int(432/2),int(368/2)# w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h),
                        tf_config=config,manual_roi=(0,0))

    if args.video =='webcam':
        args.video = 0
    else:
        args.video = args.video
    cap = cv2.VideoCapture(args.video)

    #Video Writer Parameter
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    ori_w = int(cap.get(3))
    ori_h = int(cap.get(4))
    channel= 3
    background = np.zeros((ori_h,ori_w,channel))
    output_fps = 25

    output_file = (args.video).split('/')[-1].split('.')[0] + '.avi'
    out = cv2.VideoWriter(output_file,fourcc, output_fps, (int(cap.get(3)), int(cap.get(4))))

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    i = 0
    if args.roi_cut == 'unset':
        x0,x1,y0,y1 = 0,0,ori_w,ori_h
    else:
        roi_cut = args.roi_cut.split(',')
        x0,y0,x1,y1 = int(roi_cut[0]),int(roi_cut[1]),int(roi_cut[2]),int(roi_cut[3])
    print(x0,y0,x1,y1)
    while cap.isOpened():
        ret_val, image = cap.read()
        if ret_val==True:
            image_ori = image.copy()
            image = image[x0:y0,x1:y1] # ROI Image
            if i<1:
                asa = stabilkan(image,0,10)
            else:
                dist = asa.distance_calc(image)
                asa.update(dist,image)
                image = asa.goodframe
                # print(dist)
            # start_pos = [474,551]
            start_pos = [x0,x1]
            humans = e.inference(image,resize_to_default=True, upsample_size=6.0)
            image_shape = image.shape[:2]
            if i<1:
                #inisialize
                point = stabilkanpointd(humans.copy(),image_shape,0,start_pos)
            else:
                #update
                point.update(humans.copy(),i)
                distance = point.dis_cal()

            image = TfPoseEstimator.draw_humans(image_ori,image,start_pos, humans, imgcopy=False)

            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10)
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)

            out.write(image)
            fps_time = time.time()
            i +=1

            # Stopping Method
            if cv2.waitKey(1) == 27:
                saver_json(point,ori_w,ori_h,args.video)
                break
        else:
            saver_json(point,ori_w,ori_h,args.video)
            break
    cv2.destroyAllWindows()
    out.release()
    cap.release()

logger.debug('finished+')
