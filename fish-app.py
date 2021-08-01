# Importing Required Modules
from keras.saving.saved_model.load import load
import streamlit as st
import numpy as np
import os
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave, imshow
from skimage.transform import resize
import time
from urllib.request import urlretrieve

from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img, array_to_img

# Layout of app
st.title("Fish Colorization Using Autoencoder")

st.info("This Web Application works with a pretrained **Autoencoder** which will feed on a *GrayScale Image* and return a *RGB Image*. The colorization of images has an accuracy of about **90.0%**.\
    You can either **Upload your image** or **Use some sample images** or **Provide an image address**")

# Loading pretrained model
model = load_model("fish_model.h5")

def prediction(img, model):
    # Original Model Result
    img1_color=[]

    img1 = img_to_array(img)
    img1 = resize(img1 ,(256, 256))

    img1_color.append(img1)

    img1_color = np.array(img1_color, dtype=float)
    img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
    img1_color = img1_color.reshape(img1_color.shape+(1,))

    output1 = model.predict(img1_color)
    output1 = output1*128

    result = np.zeros((256, 256, 3))
    result[:,:,0] = img1_color[0][:,:,0]
    result[:,:,1:] = output1[0]

    result = lab2rgb(result)

    return result

def process_image(img):
    try:
        # Original 3D image from user
        img.save("image.png")
        org_img = img_to_array(img)

        # Input to the model
        inp_image = load_img("image.png")
        inp_img = img_to_array(inp_image)
        
        # 3D Image as the Input Image
        user_inp_img = resize(inp_img ,(256, 256))
        user_inp_img = resize(rgb2lab(1.0/255 * user_inp_img)[:,:,0], (256, 256, 1))

        # User's Original Image
        r1, r2, r3 = st.beta_columns(3)
        r2.markdown("# Original Image")
        r1, r2, r3 = st.beta_columns(3)
        r2.image(array_to_img(org_img), width=300)

        # Image taken as the Input
        r1, r2, r3 = st.beta_columns(3)
        r2.markdown("# Image taken as input")
        r1, r2, r3 = st.beta_columns(3)
        r2.image(array_to_img(user_inp_img), width=300)

        # Prediction
        pred = prediction(inp_img, model)

        predicting = st.warning("Processing your output...")
        my_bar = st.progress(0)

        for percent_complete in range(100):
            time.sleep(0.085)
            my_bar.progress(percent_complete + 1)
        
        my_bar.empty()
        predicting.empty()

        suc = st.success(f"Image is successfully processed")

        predicted_img = array_to_img(resize(pred, org_img.shape))

        # Predicted Image / Colored Image
        r1, r2, r3 = st.beta_columns(3)
        r2.markdown("# Colored Image")
        r1, r2, r3 = st.beta_columns(3)
        r2.image(predicted_img, width=300)

    except Exception as e:
        st.error("Image cannott be processed")


# Sample images
sample_img =  st.checkbox("Check for sample images")


if sample_img:
    images = st.sidebar.selectbox("Select a sample image", os.listdir("sample fish images"))

    img = Image.open(os.path.join("sample fish images", images))

    process_image(img)

# Uploading images
UPL_HEAD = st.subheader("Upload the Image of Fish")
UPL_IMG= st.file_uploader(label="Upload Image", type=["png", "jpg"], help="Use Images with extenstions as **'png'** and **'jpg'**")


if UPL_IMG is not None:
    try:
        # Changing Layout for Uploader
        img = Image.open(UPL_IMG)

        process_image(img)
    
    except Exception as e:
        st.error("Image cannot be processed. Use images with extension as **'png'** or **'jpg'**")

# Using link
IMG_LINK_HEAD = st.subheader("Paste the link of Image")
IMG_LINK = st.text_input("Link for Image", value=" ", help="Use the link which has Image with extensions as **'png'** and **'jpg'**")

if IMG_LINK is not " ":
    try:
        # Changin Layout for image link
        urlretrieve(IMG_LINK, "image.png")
        
        img = Image.open('image.png')

        process_image(img)
    except Exception as e:
        st.error("Image cannot be processed. Try with other files")