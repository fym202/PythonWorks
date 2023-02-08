# -*- coding: utf-8 -*-
"""
Created on 2022-07-12 22:54

@author: Fan yi ming

Func: runer
"""
import numpy as np
import cv2
from openvino.inference_engine import IECore
import os
# 参数1
emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
genders = ['female', 'male']


# 创建推理引擎
#创建推理引擎类
class InferenceEngineopenVINO:
    #输入IR文件中.xml文件的路径,以及推理设备
    def _init__(self,net_model_xml_path,device="CPU" , img_size=(360,360)):
        self.device =device
        self.img_size = img_size
        net_model_bin_path = os.path.splitext(net_model_xml_path)[0] + '.bin'#1创建推理引擎核心对象
        # 1创建推理引擎
        self.ie = IECore()
        # 2读取中间表示文件(.xml ,.bin文件)
        self.net = self.ie.read_network(model=net_model_xml_path,weights=net_model_bin_path)
        # 3将网络加载到特定的设备
        self.exec_net = self.ie.load_network(network=self.net,num_requests=1,device_name=device)

    #定义推理方法
    def infer(self, img=None) :
        # 4设置输入数据,根据网络输入的格式对输入图片进行缩放input_layer = next(iter(self.net.inputs))
        n, c, h,w = self.net.inputs[input_layer].shapeimage_resized = cv2.resize(img, self.img_size)image_np = image_resized / 255.e#归一化到e~1
        img = np.transpose( image_np, (2,e,1))[None,]print( "img_", img.shape)
        #5使用指定设备进行推理
        inference_result = self.exec_net.infer(inputs={input_layer: img})
        #6返回推理结果
        return inference_result

cap = cv2.videocapture(0)
inf_openvino = InferenceEngineopenVINO( "./models/face-detection-0200.xml" , device="CPU")

while True:
    ret,frame = cap.read()
    if not ret:
        continue
    print(inf_openvino.infer(frame))
    cv2.imshow("img", frame)
    cv2.waitKey(10)
