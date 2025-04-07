#!/usr/bin/env python
# coding: utf-8

# In[134]:


import zipfile
from PIL import Image
import io
import matplotlib.pyplot as plt


# In[135]:


import gzip
import zipfile
import os


with gzip.open(r"C:\Users\surya\Downloads\trainimages.gz", 'rb') as f:
    file_content = f.readline()
    
    


# In[136]:


with gzip.open(r"C:\Users\surya\Downloads\trainlabels.gz", 'rb') as f1:
    file_content1 = f1.read()
    


# In[137]:


with gzip.open(r"C:\Users\surya\Downloads\testimages.gz", 'rb') as f2:
    file_content2 = f2.readline()
    


# In[138]:


with gzip.open(r"C:\Users\surya\Downloads\testlabels.gz", 'rb') as f3:
    file_content3 = f3.read()


# In[139]:


import pandas as pd


# In[140]:


import torch
import torchvision


# In[141]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[142]:


from torch.autograd import Variable


# In[143]:


import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Instantiate the model
input_size = 1008
hidden_size = 512
num_classes = 2
model = SimpleNet(input_size, hidden_size, num_classes)
model


# In[144]:


import numpy as np


# In[145]:


numpy_array = np.frombuffer(file_content, dtype=np.uint8)
numpy_array


# In[146]:


numpy_array1 = np.frombuffer(file_content1, dtype=np.uint8)
numpy_array1


# In[147]:


numpy_array2 = np.frombuffer(file_content2, dtype=np.uint8)
numpy_array2


# In[148]:


numpy_array3 = np.frombuffer(file_content3, dtype=np.uint8)
numpy_array3


# In[149]:


tensor_from_int = torch.tensor(numpy_array)
ff = tensor_from_int.float()
ff


# In[150]:


tensor_from_int1 = torch.tensor(numpy_array1)
ff1= tensor_from_int1.float()
ff1


# In[151]:


tensor_from_int2 = torch.tensor(numpy_array2)
ff2= tensor_from_int2.float()
ff2


# In[152]:


tensor_from_int3 = torch.tensor(numpy_array3)
ff3= tensor_from_int3.float()
ff3


# In[153]:


ib = ff.unsqueeze(0)
ib=ib.unsqueeze(0)
ib


# In[154]:


ib1 = ff1.unsqueeze(0)
ib1=ib1.unsqueeze(0)
ib1


# In[155]:


ib2 = ff2.unsqueeze(0)
ib2=ib2.unsqueeze(0)
ib2


# In[156]:


ib3 = ff3.unsqueeze(0)
ib3=ib3.unsqueeze(0)
ib3


# In[157]:



# In[127]:
outputs = model(ib)
print(outputs)

predicted_index1 = torch.argmax(outputs).sum().item()
print(predicted_index1)



# In[158]:


from streamlit_drawable_canvas import st_canvas


# In[159]:
import streamlit as st 
import pandas as pd
import numpy as np
import requests

import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image

canvas_result = st_canvas(drawing_mode="freedraw")
number = st.number_input("Insert a number")
st.text_area(label="Output Data:", value=number, height=100)
st.text_area(label="Predictions:", value=outputs, height=100)
st.text_area(label="Prediction Index:", value=predicted_index1, height=100)


import logging
import streamlit as st
import os
from datetime import datetime

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure the logger
log_file_path = os.path.join('logs', f'app_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

st.title("Logging")

if st.button("Log an event"):
    logger.info("Button clicked!")
    st.success(log_file_path)
