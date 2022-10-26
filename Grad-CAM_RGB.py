@NathanWit

# ## **1. Import libraries**

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import cv2


# #### **Displaying the image we'll use as example**

# In[2]:


img_path = r'C:\Users\witkowicz\Desktop\Stage Nathan WITKOWICZ\Image\Noise_32.jpg'

img = keras.preprocessing.image.load_img(img_path,color_mode='rgb')


# ## **2. The Grad-CAM algorithm**

# #### **Converting a PIL image to an a resized array**

# In[3]:


def get_img_array(img_path, size):

    img = keras.preprocessing.image.load_img(img_path,color_mode='rgb', target_size=size) # `img` is a PIL (Python Imaging Library) image of size 299x299
    array = keras.preprocessing.image.img_to_array(img)  # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = np.expand_dims(array, axis=0) # We add a dimension to transform our array into a "batch" of size (1, 299, 299, 3)

    return array


# #### **Generate class activation heatmap**

# In[4]:


def make_gradcam_heatmap(img, model, last_conv_layer_name, pred_index=None, plot_activation=True):

    # First, we create a model that maps the input image to the activations of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image with respect to the activations of the last conv layer
    with tf.GradientTape(persistent=True) as tape: # persitent = True indicates that we allow more than one gradient method call
        img = tf.convert_to_tensor(img)
        tape.watch( img ) # We start the recording/tracing of our tensor 'img'
        last_conv_layer_output, preds = grad_model(img, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        print(class_channel)



    # Gradient of the output neuron (top predicted or chosen) with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_sum(grads, axis=(0, 1, 2)) + np.finfo(np.float32).eps # Adding an epsilon to avoid dividing by zero (minimum numerical value according to processor)

    # We multiply each channel in the feature map array by "how important this channel is" with regard to the top predicted class then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis] # '@' corresponds to the matrix product
    heatmap = tf.squeeze(heatmap) # Removes dimensions of size 1 from the shape of a tensor

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # For the subplot of each feauture maps of the layer of interest/activation

    if plot_activation:
        k = last_conv_layer_output.shape[-1] # Number of feature maps
        fig, axs = plt.subplots(nrows=int(k/4), ncols=4, figsize=(20,15))
        plt.suptitle(f'Layer activation: {last_conv_layer_name}', y=0.90, fontweight='bold')

        plt.subplots_adjust(hspace=0.5)
        for i, ax in enumerate(axs.flat):
            ax.imshow(last_conv_layer_output[:,:,i], cmap='jet')
            ax.set_title(i)
        plt.show()


    return heatmap.numpy()


# In[5]:


def make_gradcam_heatmap_voriginal(img_array, model, last_conv_layer_name, pred_index=None):

    # First, we create a model that maps the input image to the activations of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output neuron (top predicted or chosen) with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    """
    We multiply each channel in the feature map array by "how important this channel is" with regard to the top predicted class then sum all the channels to obtain the heatmap class
    activation
    """
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis] # '@' corresponds to the matrix product
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


# ## **3. Superimposed visualization**

# In[6]:


def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


# ## **4. Let's test-drive it**

# In[7]:


def test_fonction(img_path):
    img_size = (50, 500,3)

    img_original = keras.preprocessing.image.load_img(img_path, color_mode='rgb')

    preprocess_input = keras.applications.xception.preprocess_input
    img_array = preprocess_input(get_img_array(img_path, size=img_size)) #adequate your image to the format the model requires


    model_path_binary = r"C:\Users\witkowicz\Desktop\Stage Nathan WITKOWICZ\CNN/my_model.h5" #Old model without 2 and Categ
    model = models.load_model(model_path_binary) # Loading the pre-trained model we'll use and try to explain with Grad-CA)
    model.layers[-1].activation = None # Remove last layer's softmax

    predictions = model.predict(img_array)

    last_conv_layer_name = "conv2d"
    img_0 = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    last_conv_layer_name = "conv2d_2"
    img_2 = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    return img_original, img_0, img_2


# #### **Choosing our image's dimensions**

# In[8]:


img_size = (50, 500,3)


# #### **Resizing and converting the image into an array**

# In[9]:


"""
1. Preprocesses a tensor or Numpy array encoding a batch of images.
"""
preprocess_input = keras.applications.xception.preprocess_input
img_array = preprocess_input(get_img_array(img_path, size=img_size)) #adequate your image to the format the model requires


# In[10]:


"""
2. Instanciation of a model with pre-trained weights : model.summary()
"""
model_path_binary = r"C:\Users\witkowicz\Desktop\Stage Nathan WITKOWICZ\CNN/my_model.h5" #Old model without 2 and Categ

model = models.load_model(model_path_binary) # Loading the pre-trained model we'll use and try to explain with Grad-CA)
model.layers[-1].activation = None # Remove last layer's softmax


# #### **Identifying and printing what the top predicted class is**



"""
3. Decodes the prediction of an ImageNet model.
"""
predictions = model.predict(img_array)
predictions


'''
Generate class activation heatmap and displaying it
'''


"""
4. Selecting the layer we'll use for our heatmap
"""
last_conv_layer_name = "conv2d_2"
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
plt.imshow(heatmap, cmap='jet')
plt.title(f'Final heatmap of layer: {last_conv_layer_name}', fontweight='bold')
plt.colorbar()
plt.show()


'''
Create a superimposed visualization and saved it
'''

save_and_display_gradcam(img_path, heatmap, cam_path="Grad-cam-test_conv2d.jpg")


plt.imshow(img)


heatmap_conv2d          = make_gradcam_heatmap(img_array, model, 'conv2d',        plot_activation=False)
heatmap_max_pooling2d   = make_gradcam_heatmap(img_array, model, 'max_pooling2d', plot_activation=False)
heatmap_conv2d_1        = make_gradcam_heatmap(img_array, model, 'conv2d_1',        plot_activation=False)
heatmap_max_pooling2d_1 = make_gradcam_heatmap(img_array, model, 'max_pooling2d_1', plot_activation=False)
heatmap_conv2d_2        = make_gradcam_heatmap(img_array, model, 'conv2d_2',        plot_activation=False)
heatmap_max_pooling2d_2 = make_gradcam_heatmap(img_array, model, 'max_pooling2d_2', plot_activation=False)



plt.subplot(321),plt.imshow(heatmap_conv2d,       'jet'),   plt.title('conv2d',       fontweight='bold'),   plt.colorbar()
plt.subplot(322),plt.imshow(heatmap_max_pooling2d,'jet'),   plt.title('max_pooling2d',fontweight='bold'),   plt.colorbar()
plt.subplot(323),plt.imshow(heatmap_conv2d_1,       'jet'), plt.title('conv2d_1',       fontweight='bold'), plt.colorbar()
plt.subplot(324),plt.imshow(heatmap_max_pooling2d_1,'jet'), plt.title('max_pooling2d_1',fontweight='bold'), plt.colorbar()
plt.subplot(325),plt.imshow(heatmap_conv2d_2,       'jet'), plt.title('conv2d_2',       fontweight='bold'), plt.colorbar()
plt.subplot(326),plt.imshow(heatmap_max_pooling2d_2,'jet'), plt.title('max_pooling2d_2',fontweight='bold'), plt.colorbar()
fig = plt.gcf()
fig.set_size_inches(16, 12)


'''
II- Exemple sur 1 image
'''


image_Original, image_0, image_2 = test_fonction(r'C:\Users\witkowicz\Desktop\Stage Nathan WITKOWICZ\Image\FR_32.jpg')


ax1 = plt.subplot(311)
plt.imshow(image_Original,cmap='jet')
plt.title('Version original', fontweight='bold')

ax2 = plt.subplot(312)
plt.imshow(image_0,cmap='jet')
plt.title('Version Grad Cam conv2d', fontweight='bold')

ax3 = plt.subplot(313)
plt.imshow(image_2,cmap='jet')
plt.title('Version Grad Cam conv2d_2', fontweight='bold')

plt.show()
