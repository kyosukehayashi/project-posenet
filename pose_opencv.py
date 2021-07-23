# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from functools import partial
import re
import time
import os
import enum

import numpy as np
from PIL import Image
import cv2
# import svgwrite
# import gstreamer

from pose_engine import PoseEngine,KeypointType

class KeypointType(enum.IntEnum):
    """Pose kepoints."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

EDGES = (
    (KeypointType(0), KeypointType(1)),
    (KeypointType(0), KeypointType(2)),
    (KeypointType(0), KeypointType(3)),
    (KeypointType(0), KeypointType(4)),
    (KeypointType(3), KeypointType(1)),
    (KeypointType(4), KeypointType(2)),
    (KeypointType(1), KeypointType(2)),
    (KeypointType(5), KeypointType(6)),
    (KeypointType(5), KeypointType(7)),
    (KeypointType(5), KeypointType(11)),
    (KeypointType(6), KeypointType(8)),
    (KeypointType(6), KeypointType(12)),
    (KeypointType(7), KeypointType(9)),
    (KeypointType(8), KeypointType(10)),
    (KeypointType(11), KeypointType(12)),
    (KeypointType(11), KeypointType(13)),
    (KeypointType(12), KeypointType(14)),
    (KeypointType(13), KeypointType(15)),
    (KeypointType(14), KeypointType(16)),
)


def shadow_text(img, x, y, text, font_size=0.5):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,255), 1, cv2.LINE_AA)


def draw_pose(img, pose, threshold=0.2):
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        print(label)
        if(label == 0):color = (0,0,0)
        elif(label%2==0):color = (0,255,0)# right green
        else:color=(255,0,0)#left  blue
        '''
        print(keypoint)
        print(keypoint.point)
        print(keypoint.point[0])
        print(keypoint.point[1])
        '''
        xys[label] = (keypoint.point[0], keypoint.point[1])
        img = cv2.circle(img, (keypoint.point[0], keypoint.point[1]), 4, color, -1)

    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        img = cv2.line(img, (ax, ay), (bx, by), (0, 255, 255), 2)
    return xys

def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]

def calc_dir(shoulder,view_point):
    """
    shoulder:float (2,2) [[position of left shoulder],[position of right shoulder]]
    view_point:float (2) [[position of view_point]
    """
    vec_shoulder = np.array(shoulder[0]) - np.array(shoulder[1])
    vec_sight = np.array(view_point) - (np.array(shoulder[0]) + np.array(shoulder[1]))/2
    size_shoulder = np.linalg.norm(vec_shoulder, ord=2)
    size_sight = np.linalg.norm(vec_sight, ord=2)
    cos_ = np.inner(vec_shoulder,vec_sight)/(size_shoulder*size_sight)
    theta = np.arccos(cos_)*180/np.pi
    det = determinant(vec_shoulder,vec_sight)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return theta
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-theta

def draw_viewline(img, pos,_view_x=0,_view_y=0,angle_range=45):
    left_shoulder = [pos[5][0],pos[5][1]]
    print(left_shoulder)
    right_shoulder = [pos[6][0],pos[6][1]]
    print(right_shoulder)
    print([left_shoulder[0:2],right_shoulder[0:2]])
    view_x = _view_x*img.shape[1]
    view_y = _view_y*img.shape[0]
    print("left_shoulder")
    print(left_shoulder[0:2])
    print("right_shoulder")
    print(right_shoulder[0:2])
    angle1 = calc_dir([right_shoulder[0:2],left_shoulder[0:2]],[view_x,view_y])
    print(angle1)
    pos1 = (int(0.5*(left_shoulder[0]+right_shoulder[0])), int(0.5*(left_shoulder[1]+right_shoulder[1])))
    pos2 = (int(view_x), int(view_y))
    if(np.abs(angle1-90) < angle_range):
      #num_watching_viewpoint[i] += 1
      cv2.line(img,pos1,pos2,(0, 0, 255))
      return 1
    else:
      cv2.line(img,pos1,pos2,(255, 0, 0))
      return 0

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use (WebCam number or video file path)', default='0')
    parser.add_argument('--imgsrc',default='0')
    parser.add_argument('--view-point-x', type=float, default=0, help='x value of view point')
    parser.add_argument('--view-point-y', type=float, default=0, help='y value of view point')
    parser.add_argument('--angle-range', type=float, default=45, help='permitted angle range')
   
    # parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    args = parser.parse_args()
    default_model = 'models/mobilenet/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

    print('Loading model: ', model)
    engine = PoseEngine(model, mirror=args.mirror)
    # engine = PoseEngine(model)


    last_time = time.monotonic()
    n = 0
    sum_fps = 0
    sum_process_time = 0
    sum_inference_time = 0

    width, height = src_size

    isVideoFile = False
    isImgFile = False
    frameCount = 0
    maxFrames = 0
    
    # VideoCapture init
    videosrc = args.videosrc
    imgsrc = args.imgsrc
    if not videosrc.isdigit():
        isVideoFile = os.path.exists(videosrc)
    elif not imgsrc.isdigit():
        isImgFile = os.path.exists(imgsrc)
    else:
        videosrc = int(videosrc)
    if not isImgFile:
        print("Start VideoCapture")
        cap = cv2.VideoCapture(videosrc)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) 
        if cap.isOpened() == False:
            print('can\'t open video source \"%s\"' % str(videosrc))
            return;

        print("Open Video Source")
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if isVideoFile:
            maxFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        output_video_name='./output/{}_{}'.format(args.res,videosrc.split('/')[-1])
        print(output_video_name)
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
        writer = cv2.VideoWriter(output_video_name, fmt, frame_rate, (appsink_size[0],appsink_size[1]))
    else:
        maxFrames = 1
        img = cv2.imread(imgsrc)
        img = cv2.resize(img , (width, height))
        output_img_name='./output/{}_{}'.format(args.res,imgsrc.split('/')[-1])

    try:
        while (frameCount < maxFrames):
            if isImgFile:
                frame = img
            else: 
                ret, frame = cap.read()
                if ret == False:
                    print('can\'t read video source')
                    break;
            print(type(frame))
            print(frame.size)
            print(frame.shape)
            rgb = frame[:,:,::-1]

#             nonlocal n, sum_fps, sum_process_time, sum_inference_time, last_time
            start_time = time.monotonic()
            # image = Image.fromarray(rgb)
            outputs, inference_time= engine.DetectPosesInImage(rgb)
            end_time = time.monotonic()
            n += 1
            sum_fps += 1.0 / (end_time - last_time)
            sum_process_time += 1000 * (end_time - start_time) - inference_time
            sum_inference_time += inference_time
            last_time = end_time
            text_line = 'PoseNet: %.1fms Frame IO: %.2fms TrueFPS: %.2f Nposes %d' % (
                sum_inference_time / n, sum_process_time / n, sum_fps / n, len(outputs)
            )
            print(text_line)

            # crop image
            #cv2.imwrite('test.jpg',frame)
            imgDisp = cv2.resize(frame,(appsink_size[0],appsink_size[1]))
            #imgDisp = frame[0:appsink_size[1], 0:appsink_size[0]].copy()
            #cv2.imwrite('test2.jpg',imgDisp)
            if args.mirror == True:
                imgDisp = cv2.flip(imgDisp, 1)

            shadow_text(imgDisp, 10, 20, text_line)
            for pose in outputs:
                xys_result = draw_pose(imgDisp, pose)
                #check if watching viewpoint
                # print(args.view_point_x)
                # print(imgDisp.shape[0])
                print("xys_result.keys()")
                print(xys_result)
                print(xys_result.keys())
                # print(xys_result[6])
                # print(xys_result[5])
                print(pose.keypoints)
                print(type(pose.keypoints.keys()))
                print('leftshoulder check')
                print(KeypointType(5))
                print(pose.keypoints.keys())
                if((KeypointType(5) in xys_result.keys()) and (KeypointType(6) in xys_result.keys())):
                    draw_viewline(imgDisp,xys_result,args.view_point_x,args.view_point_y,args.angle_range)

            cv2.imshow('PoseNet - OpenCV', imgDisp)
            if isImgFile:
                cv2.imwrite(output_img_name, imgDisp)
                break
            else:
                writer.write(imgDisp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif isVideoFile:
                    frameCount += 1
                # # check frame count
                # if frameCount >= maxFrames:
                #     # rewind video file
                #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                #     frameCount = 0


    except Exception as ex:
        raise ex
    finally:
        cv2.destroyAllWindows()
        if not isImgFile:
            cap.release()
            writer.release()
            print(output_video_name)
            print(appsink_size)



if __name__ == '__main__':
    main()