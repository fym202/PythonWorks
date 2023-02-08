# -*- coding: utf-8 -*-
"""
Created on 2022-07-12 23:45

@author: Fan yi ming

Func:
"""
import numpy as np
import time
import cv2 as cv
from openvino.inference_engine import IECore
emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
genders = ['female', 'male']


def face_landmark_demo():
    ie = IECore()
    for device in ie.available_devices:
        print(device)

    model_xml = "D:/projects/models/face-detection-0102/FP32/face-detection-0102.xml"
    model_bin = "D:/projects/models/face-detection-0102/FP32/face-detection-0102.bin"

    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n, c, h, w)

    cap = cv.VideoCapture("D:/images/video/example_dsh.mp4")
    exec_net = ie.load_network(network=net, device_name="CPU")

    # 鍔犺浇浜鸿劯琛ㄦ儏璇嗗埆妯″瀷
    em_xml = "D:/projects/models/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml"
    em_bin = "D:/projects/models/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.bin"

    em_net = ie.read_network(model=em_xml, weights=em_bin)
    em_input_blob = next(iter(em_net.input_info))
    em_out_blob = next(iter(em_net.outputs))
    en, ec, eh, ew = em_net.input_info[em_input_blob].input_data.shape
    print(en, ec, eh, ew)

    em_exec_net = ie.load_network(network=em_net, device_name="CPU")

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        image = cv.resize(frame, (w, h))
        image = image.transpose(2, 0, 1)
        inf_start = time.time()
        res = exec_net.infer(inputs={input_blob: [image]})
        inf_end = time.time() - inf_start
        # print("infer time(ms)锛�%.3f"%(inf_end*1000))
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
                rh, rw, rc = roi.shape
                roi_img = cv.resize(roi, (ew, eh))
                roi_img = roi_img.transpose(2, 0, 1)
                em_res = em_exec_net.infer(inputs={em_input_blob: [roi_img]})
                prob_landmarks = em_res[em_out_blob]
                for index in range(0, len(prob_landmarks[0]), 2):
                    x = np.int(prob_landmarks[0][index] * rw)
                    y = np.int(prob_landmarks[0][index+1] * rh)
                    cv.circle(roi, (x, y), 3, (0, 0, 255), -1, 8, 0)
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
                cv.putText(frame, "infer time(ms): %.3f" % (inf_end * 1000), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (255, 0, 255),
                           2, 8)
        cv.imshow("Face+emotion Detection", frame)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.waitKey(0)
    cv.destroyAllWindows()


def face_emotion_demo():
    ie = IECore()
    for device in ie.available_devices:
        print(device)

    model_xml = "D:/projects/models/face-detection-0102/FP32/face-detection-0102.xml"
    model_bin = "D:/projects/models/face-detection-0102/FP32/face-detection-0102.bin"

    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n, c, h, w)

    cap = cv.VideoCapture("D:/images/video/example_dsh.mp4")
    exec_net = ie.load_network(network=net, device_name="CPU")

    #
    em_xml = "D:/projects/models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml"
    em_bin = "D:/projects/models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin"

    em_net = ie.read_network(model=em_xml, weights=em_bin)

    em_input_blob = next(iter(em_net.input_info))
    em_out_blob = next(iter(em_net.outputs))
    en, ec, eh, ew = em_net.input_info[em_input_blob].input_data.shape
    print(en, ec, eh, ew)

    em_exec_net = ie.load_network(network=em_net, device_name="CPU")

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        image = cv.resize(frame, (w, h))
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
        cv.imshow("Face+emotion Detection", frame)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.waitKey(0)
    cv.destroyAllWindows()


def face_age_gender_demo():
    # 1创建推理引擎
    ie = IECore()
    for device in ie.available_devices:
        print(device)

    # 模型位置信息
    model_xml = "./models/face-detection/face-detection-0200.xml"
    model_bin = "./models/face-detection/face-detection-0200.bin"
    # 2读取中间表示文件(.xml ,.bin文件)
    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n, c, h, w)

    cap = cv.VideoCapture("D:/images/video/example_dsh.mp4")
    exec_net = ie.load_network(network=net, device_name="CPU")

    # 鍔犺浇浜鸿劯琛ㄦ儏璇嗗埆妯″瀷
    em_xml = "D:/projects/models/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml"
    em_bin = "D:/projects/models/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin"

    em_net = ie.read_network(model=em_xml, weights=em_bin)
    em_input_blob = next(iter(em_net.input_info))
    em_it = iter(em_net.outputs)
    em_out_blob1 = next(em_it)
    em_out_blob2 = next(em_it)
    en, ec, eh, ew = em_net.input_info[em_input_blob].input_data.shape
    print(en, ec, eh, ew)

    em_exec_net = ie.load_network(network=em_net, device_name="CPU")

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        image = cv.resize(frame, (w, h))
        image = image.transpose(2, 0, 1)
        inf_start = time.time()
        res = exec_net.infer(inputs={input_blob:[image]})
        inf_end = time.time() - inf_start
        # print("infer time(ms)锛�%.3f"%(inf_end*1000))
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
                roi = frame[ymin:ymax,xmin:xmax,:]
                roi_img = cv.resize(roi, (ew, eh))
                roi_img = roi_img.transpose(2, 0, 1)
                em_res = em_exec_net.infer(inputs={em_input_blob: [roi_img]})
                age_conv3 = em_res[em_out_blob1].reshape(1, 1)[0][0] * 100
                prob_age = em_res[em_out_blob2].reshape(1, 2)
                label_index = np.int(np.argmax(prob_age, 1))
                age = np.int(age_conv3)
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
                cv.putText(frame, "infer time(ms): %.3f"%(inf_end*1000), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255),
                           2, 8)
                cv.putText(frame, genders[label_index] + ', ' +str(age), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255),
                          2, 8)
        cv.imshow("Face+emotion Detection", frame)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    face_emotion_demo()
