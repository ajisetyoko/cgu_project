import numpy as np
import json
def perframe_process(list_bodyparts,frame_number):
    list_human = {}
    list_human['Frame Number'] = frame_number

    for human_number in range(len(list_bodyparts)):
        list_human['Human Confidence Score'] = list_bodyparts[human_number].score
        list_human['Detection no '+str(human_number)] = bodypart_tolist(list_bodyparts[human_number])
    return list_human
def bodypart_tolist(bodyparts):
    y = []
    for part in range(17):

        try:
            parts = ({
                'name'    : str(bodyparts.body_parts[part].get_part_name()),
                'id_part' : part,
                'position': (bodyparts.body_parts[part].x,bodyparts.body_parts[part].y),
                'score'   : bodyparts.body_parts[part].score
            })
        except KeyError:
                parts = ({
                    'name'    : None,
                    'id_part' : part,
                    'position': None,
                    'score'   : None
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
def saver_json(point,ori_w,ori_h):
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

    json_file_name = 'Detection_result_json_'+'video_name'+ '.json'
    with open(json_file_name, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    # np.save('points.npy',point) for debuging
    print('Detection saved in JSON Format')
    return True
import argparse
def parser_init():
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='/media/simslab-cs/A/videoset/cgutest.mp4')
    parser.add_argument('--resolution', type=str, default='432x368',
                        help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--resize-out-ratio', type=float, default=2.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--dualCam', type = bool, default=True, help='If want to feed the model from 2 camera')
    parser.add_argument('--tracking',type = bool, default=False, help='Track human detection if there are more than 1')
    parser.add_argument('--vid_mode',type = bool, default=True, help='Feed the model from video file')
    args = parser.parse_args()

    return args

def dual_cam_init(cam_mod):
    if cam_mod == 1:
        video_capture_0 = cv2.VideoCapture(2)
        video_capture_0.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        video_capture_0.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        video_capture_0.set(cv2.CAP_PROP_FPS,15)
        return video_capture_0
    elif cam_mod==2:
        video_capture_1 = cv2.VideoCapture(0)
        video_capture_1.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        video_capture_1.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        video_capture_1.set(cv2.CAP_PROP_FPS,15)
        return video_capture_1
    elif cam_mod ==3:
        try:
            return dual_cam_init(1)
        except Error:
            return dual_cam_init(2)

def read_frame():
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()
    return ret0, frame0,ret1, frame1
