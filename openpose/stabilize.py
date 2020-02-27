# @Author: ajisetyoko <simslab-cs>
# @Date:   2020-02-25T17:31:20+08:00
# @Email:  aji.setyoko.second@gmail.com
# @Last modified by:   simslab-cs
# @Last modified time: 2020-02-27T12:11:12+08:00



import numpy as np
import cv2
from tf_pose.estimator import Human

def get_first_frame(videopath):
    cap = cv2.VideoCapture(videopath)
    cap.set(1,2)
    ret, frame = cap.read()
    return  frame

class stabilkan:
    def __init__(self,image,dist=0,dist_thres = 0.0):
        self.goodframe = image
        self.dist = dist
        self.dist_thres = dist_thres

    def distance_calc(self,new_image):
        dist = np.sum((new_image-self.goodframe)**2)
        shap = new_image.shape[0]*new_image.shape[1]
        dist = dist/shap
        self.dist = dist
        return  dist

    def update(self,dist,new_image):
        if dist<self.dist_thres:
            pass
            # print('Image same')
            # self.goodframe = new_image
        else:
            # update self.goodframe
            # print('Image is Different --> Update')
            self.goodframe = new_image

class stabilkanpointd:
    def __init__(self,humans,image_shape,frame_num,start_pos):
        self.image_h, self.image_w = image_shape
        self.w = image_shape[0]
        self.h = image_shape[1]
        self.list_human = [humans]
        self.frame_num = frame_num
        self.index = [[frame_num,len(humans)]]
        self.start_pos = start_pos
        self.kalman_format = []

    def camera_parsing(self):
        a=self.start_pos[1];b=self.start_pos[0]
        seq = []
        for human in self.list_human[-1]:
            dt = [self.frame_num]
            for part in range(17):
                try:
                    x = int(human.body_parts[part].x * self.image_w + a + 0.5)
                    y = int(human.body_parts[part].y * self.image_h + b + 0.5)
                except KeyError:
                    x = 0
                    y = 0
                dt.append(x)
                dt.append(y)
            seq.append(dt)
        mean_y = []
        y_p    = 0
        cam_0  = []
        cam_1  = []
        for human in seq:
            ori_y  = np.array(human[1:-1:2])
            y_p    = np.mean(ori_y[np.nonzero(ori_y)])
            if y_p >=640:
                #below 640
                # print(np.where(ori_y<=640))
                wrong_ = ((np.where((ori_y<=640)&(ori_y>0))[0])*2)+1
                wrong_x= wrong_ - 1
                human  = np.array(human)
                human[wrong_x.tolist()] = 0
                human[wrong_.tolist()]  = human[wrong_.tolist()] - 640
                cam_1.append(human)
            elif y_p<640:
                cam_0.append(human)
            mean_y.append(y_p)
        return [cam_0,cam_1]

    @staticmethod
    def get_outest_point(tracked):
        list_point = tracked
        output =[]
        for list_p in list_point:
            # x_p = list_p[1:-1:2]
            # x_p[x_p<0] = 0
            # y_p = list_p[2:-1:2]
            # y_p[y_p<0] = 0
            id  =  list_p[-1]
            # output.append([np.min(x_p[np.nonzero(x_p)]), np.min(y_p[np.nonzero(y_p)]),x_p.max(),y_p.max(),int(id)])
            output.append([list_p[2],list_p[3],int(id)])
        return output

    def getKalmanFormat(self):
        a=self.start_pos[1];b=self.start_pos[0]
        seq = []
        for human in self.list_human[-1]:
            dt = [self.frame_num]
            for part in range(17):
                try:
                    x = int(human.body_parts[part].x * self.image_w + a + 0.5)
                    y = int(human.body_parts[part].y * self.image_h + b + 0.5)
                except KeyError:
                    x = 0
                    y = 0
                dt.append(x)
                dt.append(y)
            seq.append(dt)
        self.kalman_format = seq
        return seq

    def update(self, humans,frame_num):
        self.frame_num = frame_num
        self.list_human.extend([humans])
        self.index.extend([[frame_num,len(humans)]])

    def deleteVar(self):
        self = None
        return None

    def dis_cal(self):
        #last frame = self.index[-1][0]
        value = self.calc(self.list_human[-1],self.list_human[-2])
        self.value = value
        # print(self.value)
        return value

    def calc(self,humans1,humans2):

        index = []
        for i in range(len(humans1)):
            for ii in range(len(humans2)):
                if humans1[i].part_count() == humans2[ii].part_count():
                    index.append([i,ii])
        list_err = [];list_same_point = []
        for a in index:
            same_point = 0
            humanerr = 0
            for aa in (humans1[a[0]].body_parts.keys()):
                try:
                    val1 = humans1[a[0]].body_parts.get(aa)
                    val2 = humans1[a[1]].body_parts.get(aa)
                except IndexError:
                    continue
                if val1==None or val2==None:
                    continue
                a1   = np.array([val1.x,val1.y])
                a2   = np.array([val2.x,val2.y])

                center1= (int(val1.x * self.w + 0.5), int(val1.y * self.h + 0.5))
                center2= (int(val2.x * self.w + 0.5), int(val2.y * self.h + 0.5))
                if center1==center2:
                    same_point += 1

                error    = (np.linalg.norm(a1-a2))*100
                humanerr = humanerr+error
            list_err.append(humanerr/17)
            list_same_point.append(same_point)
        self.list_err = list_err
        self.list_same_point = list_same_point
        return list_err,list_same_point
#
# #Just write every detection to file
# class writetofile:
#     def __init__(self,humans):
#         self.humans = humans
