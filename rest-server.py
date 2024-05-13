#!flask/bin/python
################################################################################################################################
#------------------------------------------------------------------------------------------------------------------------------                                                                                                                             
# This file implements the REST layer. It uses flask micro framework for server implementation. Calls from front end reaches 
# here as json and being branched out to each projects. Basic level of validation is also being done in this file. #                                                                                                                                  	       
#-------------------------------------------------------------------------------------------------------------------------------                                                                                                                              
################################################################################################################################
from flask import Flask, jsonify, abort, request, make_response, url_for,redirect, render_template
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
import os
import shutil 
import numpy as np
from search import recommend
import tarfile
from datetime import datetime
from scipy import ndimage
import re
#from scipy.misc import imsave

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
from tensorflow.python.platform import gfile
app = Flask(__name__, static_url_path = "")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
auth = HTTPBasicAuth()
pattern = r'im(\d+)\.jpg'

#==============================================================================================================================
#                                                                                                                              
#    Loading the extracted feature vectors for image retrieval                                                                 
#                                                                          						        
#                                                                                                                              
#==============================================================================================================================
extracted_features=np.zeros((10000,2048),dtype=np.float32)
with open('saved_features_recom.txt') as f:
    		for i,line in enumerate(f):
        		extracted_features[i,:]=line.split()
print("loaded extracted_features") 


#==============================================================================================================================
#                                                                                                                              
#  This function is used to do the image search/image retrieval
#                                                                                                                              
#==============================================================================================================================
@app.route('/imgUpload', methods=['GET', 'POST'])
#def allowed_file(filename):
#    return '.' in filename and \
#           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_img():
    print("image upload")
    result = 'static/result'
    image_names = []  # 用于存储搜索结果图片的文件名
    group_files = [[] for _ in range(9)]
    if not gfile.Exists(result):
          os.mkdir(result)
    shutil.rmtree(result)
 
    if request.method == 'POST' or request.method == 'GET':
        print(request.method)
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        print(file.filename)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file:# and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            inputloc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            recommend(inputloc, extracted_features)
            os.remove(inputloc)
            image_path = "/result"
            image_list =[os.path.join(image_path, file) for file in os.listdir(result)
                              if not file.startswith('.')]
            for image in image_list:
                image_names.append(os.path.basename(image))
            for i in range(9):
                filename1 = image_names[i]
                match = re.match(pattern, filename1)
                if match:# 如果匹配成功，则提取出数字部分并添加到列表中
                    number = match.group(1)
                    image_number = int(number)
                    # 定义标签文件夹路径
                    TAGS_FOLDER = 'database/tags'
                    # 修改标签文件路径获取的方式
                    folder_path = TAGS_FOLDER
                    for txt_filename in os.listdir(folder_path):
                        if txt_filename.endswith('.txt') and not txt_filename.endswith('_r1.txt'):  # 确保是 .txt 文件且不以 '_r1.txt' 结尾
                            file_path = os.path.join(folder_path, txt_filename)
                            with open(file_path, 'r') as file1:
                                for line in file1:
                                    # 使用正则表达式匹配每行中的数字部分
                                    txt_number = re.findall(r'\d+', line)
                                    # 如果该行中存在数字，则与图像文件名数字部分进行比较
                                    if txt_number:
                                        # 将txt_number中的数字字符串连接成一个整体字符串
                                        num_str = ''.join(txt_number)
                                        # 将整体字符串转换为整数
                                        txt_number1 = int(num_str)
                                        # 检查文件名中是否包含此数字
                                        if txt_number1 == image_number:
                                        # 将文件名添加到相应的组中
                                            group_files[i].append(txt_filename[:-4])
                                            break 
            
            images = {
                'image0': {
                    'url': image_list[0],
                    'labels': group_files[0],  # 添加标签信息
                    'names': extract_image_number(image_list[0])
                },
                'image1': {
                    'url': image_list[1],
                    'labels': group_files[1],  # 添加标签信息
                    'names': extract_image_number(image_list[1])
                },
                'image2': {
                    'url': image_list[2],
                    'labels': group_files[2],  # 添加标签信息
                    'names': extract_image_number(image_list[2])
                },
                'image3': {
                    'url': image_list[3],
                    'labels': group_files[3],  # 添加标签信息
                    'names': extract_image_number(image_list[3])
                },
                'image4': {
                    'url': image_list[4],
                    'labels': group_files[4],  # 添加标签信息
                    'names': extract_image_number(image_list[4])
                },
                'image5': {
                    'url': image_list[5],
                    'labels': group_files[5],  # 添加标签信息
                    'names': extract_image_number(image_list[5])
                },
                'image6': {
                    'url': image_list[6],
                    'labels': group_files[6],  # 添加标签信息
                    'names': extract_image_number(image_list[6])
                },
                'image7': {
                    'url': image_list[7],
                    'labels': group_files[7],  # 添加标签信息
                    'names': extract_image_number(image_list[7])
                },
                'image8': {
                    'url': image_list[8],
                    'labels': group_files[8],  # 添加标签信息
                    'names': extract_image_number(image_list[8])
                },
		    }				
            return jsonify(images)


def extract_image_number(def_image):
    my_filename=os.path.basename(def_image)
    match = re.match(pattern, my_filename)
    if match:  # 如果匹配成功，则提取出数字部分并返回
        return int(match.group(1))
    else:
        return None  # 如果没有匹配到，则返回 None

#==============================================================================================================================
#                                                                                                                              
#                                           Main function                                                        	            #						     									       
#  				                                                                                                
#==============================================================================================================================
@app.route("/")
def main():
    
    return render_template("main.html")   
if __name__ == '__main__':
    app.run(debug = True, host= '0.0.0.0')
