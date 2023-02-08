# -*- coding: utf-8 -*-
"""
Created on 2022-07-12 23:56

@author: Fan yi ming

Func:test file
"""
import numpy as np
import cv2 as cv
from openvino.inference_engine import IECore
import os
import time
emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

def face_emotion_demo(VedioFrom='./Video/example1.mp4'):
    # step1: 创建推理引擎
    ie = IECore()
    for device in ie.available_devices:
        print(device)
    # step2: 读取中间表示文件(.xml ,.bin文件)
    model_xml = "./models/face-detection/face-detection-0200.xml"
    model_bin = "./models/face-detection/face-detection-0200.bin"
    # step3:将网络加载到特定的设备
    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    # 打印输入输出形状
    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n, c, h, w)
    # step4: 加载视频
    cap = None
    if VedioFrom == "0":
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(VedioFrom)
    exec_net = ie.load_network(network=net, device_name="CPU")

    # 情绪识别
    em_xml = "./models/emotion-recognition/emotions-recognition-retail-0003.xml"
    em_bin = "./models/emotion-recognition/emotions-recognition-retail-0003.bin"

    em_net = ie.read_network(model=em_xml, weights=em_bin)

    em_input_blob = next(iter(em_net.input_info))
    em_out_blob = next(iter(em_net.outputs))
    en, ec, eh, ew = em_net.input_info[em_input_blob].input_data.shape
    print(en, ec, eh, ew)

    em_exec_net = ie.load_network(network=em_net, device_name="CPU")
    # step5: 运行模型
    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        image = cv.resize(frame, (w, h))  # 更改图片格式
        image = image.transpose(2, 0, 1)
        inf_start = time.time()
        res = exec_net.infer(inputs={input_blob: [image]})
        inf_end = time.time() - inf_start

        ih, iw, ic = frame.shape
        res = res[out_blob]
        for obj in res[0][0]:
            if obj[2] > 0.75:
                xmin = int(obj[3] * iw)
                ymin = int(obj[4] * ih)
                xmax = int(obj[5] * iw)
                ymax = int(obj[6] * ih)
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax >= iw:
                    xmax = iw - 1
                if ymax >= ih:
                    ymax = ih - 1
                roi = frame[ymin:ymax, xmin:xmax, :]
                roi_img = cv.resize(roi, (ew, eh))
                roi_img = roi_img.transpose(2, 0, 1)
                em_res = em_exec_net.infer(inputs={em_input_blob: [roi_img]})
                prob_emotion = em_res[em_out_blob].reshape(1, 5)
                label_index = np.argmax(prob_emotion, 1)
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
                cv.putText(frame, "infer time(ms): %.3f" % (inf_end * 1000), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (255, 0, 255),
                           2, 8)
                cv.putText(frame, emotions[np.int(label_index)], (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.55,
                           (0, 0, 255),
                           2, 8)
        cv.namedWindow("Face+emotion Detection",cv.WINDOW_NORMAL)
        cv.namedWindow("Face+emotion Detection", cv.WINDOW_KEEPRATIO)
        cv.imshow("Face+emotion Detection", frame)
        c = cv.waitKey(1)
        if c == 27:
            break
        
        cv.imwrite("./outcome/"+str(len(os.listdir("./outcome")))+".jpg", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':

    # face_emotion_demo(VedioFrom="./Video/2.mp4")
    face_emotion_demo(VedioFrom="0")