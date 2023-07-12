import torch
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
from ultralytics import YOLO
import ultralytics

model = YOLO('yolov8n.pt')
# model.train(data='TheSV.yaml',epochs=10)

st.columns(2)

col1, col2,col3 = st.columns(3)

with col1:
    st.header("Import image")
    img_file_buffer = st.file_uploader("Choose a file")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image) # if you want to pass it to OpenCV
        st.image(image, caption="The caption", use_column_width=True)


with col3:
    st.header("Retrieve Information:")
    button = st.button("Retrieve Information:")
    if button:
        st.write("run")        
        
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained']=False
        config['device'] = 'cpu'
        detector = Predictor(config)
        s = detector.predict(image)        
        
        
        st.header("Result:")
        st.write(s)
