import numpy as np
import pandas as pd
import streamlit as st
from utils import predict
import base64
import io
from PIL import Image



st.title('Peace and War Image Classifier')

navi = st.sidebar.selectbox(
    "Navigation",
    ['Background', 'Predict Images'])
st.write(navi)

if navi == 'Background':
    project_background = """
        Peace Journalism can be described as journalism that offers a more balanced perspective of war and conflict than that provided by the dominant mainstream media. For instance, peace journalism aims to construct realities from all sides, and to reveal less visible causes and effects of war and violence, such as their cost in terms of the dead and disabled, and of the destruction of social order and institutions, while refraining from dehumanizing the enemy.
        Several factors however constrain these preceding types of text or picture analysis. First, the method inevitably limits sample sizes, both in terms of the range of media outlets and the periods covered. Relying on such scenarios has become particularly problematic as the range of media outlets and citizen journalists have increased. Second, as Flaounas et al. point out, this form of analysis is labor-intensive, relying upon people to physically examine, interpret media content (texts or pictures) as war or peace oriented. More so, news media organizations now post fleeting news content on their websites, social media platforms, Twitter, and Facebook in particular.
        The current study therefore seeks to use existing conceptualizations of peace and war journalism to create a supervised machine learning image classifier trained and tested to identify war or peace-oriented pictures using war images collected from Twitter, Facebook, and websites of CNN, BBC, and Aljazeera.
    """
    st.write(project_background)

if navi == 'Predict Images':
    img = st.file_uploader("Upload Files",type=['png', 'jpeg', 'jpg'])
    if img is not None:
        image_data = img.read()
        image_bytes = bytearray(image_data)
        prediction = predict(image_bytes)
        st.write(prediction)



