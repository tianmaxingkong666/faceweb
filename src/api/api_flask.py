# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import os,shutil
import numpy as np
import json
import collections
import time
import tensorflow as tf
import argparse
import sys
import face_model
import cv2
from annoy import AnnoyIndex
import datetime
import random

SERVER_DIR_KEYS = ["/lfw/","/test/"]
SERARCH_TMP_DIR = "/tmp/search/"

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)

def dot(source_feature, target_feature):
    simlist = []
    length = len(target_feature)
    for i in range(length):
      sim = np.dot(source_feature, np.expand_dims(target_feature[i], axis=0).T)
      if sim[0][0] < 0:
        sim[0][0] = 0
      simlist.append(sim[0][0])
    return simlist



# 入库
@app.route("/storage", methods=['POST'])
def storage():
    file = request.files['file']
    filename = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))+str(random.randint(0, 100))+ '.jpg'
    filepath = os.path.join(args.file_server_image_dir, 'test', 'test', filename)
    file.save(filepath) 
    
    success = 0
    try:
        # 生成特征编码
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_img = model.get_aligned_face(image)
        feature = model.get_feature(face_img)
        
        # 特征编码入库
        feature_list.append(feature)
        
        # 特征编码索引
        feature=feature.tolist()[0]
        index.add_item(get_i(),feature)
        add_i()

        # 图片数组入库
        image_list.append(filepath)
        
    except Exception as e:
        print(str(e))
        success = 1
                
    json_result = collections.OrderedDict()
    json_result["success"] = success

    return json.dumps(json_result, cls=JsonEncoder)

# 搜索
@app.route("/search", methods=['POST'])
def search():
    file = request.files['file']
    filepath = os.path.join(SERARCH_TMP_DIR, file.filename)
    file.save(filepath)

    # 生成上传图片的特征编码
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    success = 0
    sim_image = []
    sim = []

    try:
        face_img = model.get_aligned_face(image)
        feature = model.get_feature(face_img)
        source_feature = feature
        feature = feature.tolist()[0]
        index.build(10) # 10 trees
        I = index.get_nns_by_vector(feature, 6)
        # 计算相似度
        target_feature = np.squeeze(np.array(feature_list)[I],1)
        sim = dot(source_feature, target_feature)
        _sim_image = np.array(image_list)[I].tolist()
        
        for image in _sim_image:
            for key in SERVER_DIR_KEYS:
                if key in image:
                   sim_image.append(args.file_server + key + image.split(key)[1])
        
    except Exception as e:
        print(str(e))
        success = 1
    
    json_result = collections.OrderedDict()
    json_result["success"] = success
    json_result["sim_image"] = sim_image
    json_result["sim"] = sim

    return json.dumps(json_result, cls=JsonEncoder)

def add_i():
    global i 
    i += 1

def get_i():
    global i 
    return i

def paresulte_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='/opt/insightface/models/model-r100-ii/model,00', help='path to load model.')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument('--file_server_image_dir', type=str,help='Base dir to the face image.', default='/opt/images')
    parser.add_argument('--file_server', type=str,help='the file server address', default='http://192.168.247.128:8082')
    parser.add_argument('--port', default=5000, type=int, help='api port')
    return parser.parse_args(argv)

args = paresulte_arguments('')
index = AnnoyIndex(512)
feature_list = []
image_list = [] 
model = face_model.FaceModel(args)
i = 0

if __name__ == '__main__':
    app.run(host='0.0.0.0',port = args.port, threaded=True)

