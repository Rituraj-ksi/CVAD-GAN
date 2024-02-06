#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.datasets.mnist import load_data as load_data
import random
from keras.models import Sequential, Model
from keras.layers import GaussianNoise, Conv2D, Conv2DTranspose, BatchNormalization, MaxPool2D, Flatten, Dense, Reshape, UpSampling2D, LeakyReLU, ReLU, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import tanh


# In[1]:


import scipy
from scipy import signal
#import mxnet as mx

from matplotlib import pyplot as plt
from matplotlib import colors
#from mxnet import gluon,nd
import glob
import numpy as np
import os
from PIL import Image, ImageOps

# create dataloader (batch_size, 1, 100, 100)
def create_dataset(path, batch_size, shuffle):
  files = glob.glob(path)
  data = np.zeros((len(files),1,160,160))

  for idx, filename in enumerate(files):
    im = Image.open(filename)
    #im = ImageOps.grayscale(im)
    im = im.resize((160,160))
    data[idx,0,:,:] = np.array(im, dtype=np.float32)/255.0

  #dataset = gluon.data.ArrayDataset(mx.nd.array(data, dtype=np.float32))
  #dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, last_batch="rollover", shuffle=shuffle)
  dataloader = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)

  return dataloader

# create dataloader (batch_size, 10, 227, 227)
def create_dataset_stacked_images(path, batch_size, shuffle, augment=True):
  
  files = sorted(glob.glob(path))
  if augment:
    files = files + files[2:] + files[4:] + files[6:] + files[8:]
  data = np.zeros((int(len(files)/10),10,160,160))
  i, idx = 0, 0
  for filename in files:
    im = Image.open(filename)
    im = im.resize((160,160))
    data[idx,i,:,:] = np.array(im, dtype=np.float32)/255.0
    i = i + 1
    if i > 9: 
      i = 0
      idx = idx + 1
  #dataset = gluon.data.ArrayDataset(mx.nd.array(data, dtype=np.float32))
  #dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, last_batch="rollover", shuffle=shuffle)
  dataloader = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)

  return dataloader

# perform inference and plot results
def plot_images(path, model, output_path="img", stacked=False, lstm=False):

  if not stacked:
    dataloader = create_dataset(path, batch_size=1, shuffle=False)
  else:
    dataloader = create_dataset_stacked_images(path, batch_size=1, shuffle=False, augment=False)

  counter = 0
  #model.load_parameters(params_file, ctx=ctx)
  
  try:
    os.mkdir(output_path,exist_ok=True)
  except:
    pass

  for image in dataloader:
    
    # perform inference
    #image = image.as_in_context(ctx)
    #if not lstm:
      #image_batch1=tf.transpose(image_batch,perm=[0,2,3,1])
      #reconstructed = model(image)
    #else:
      #states = model.temporal_encoder.begin_state(batch_size=1, ctx=ctx)
      #reconstructed, states = model(image, states)
    #compute difference between reconstructed image and input image 
    #reconstructed = reconstructed.asnumpy()
    #image = image.asnumpy()
    #diff = np.abs(reconstructed-image)

    if not lstm:
      image_batch1=tf.transpose(image,perm=[0,2,3,1])
      reconstructed = model(image_batch1)
    else:
      states = model.temporal_encoder.begin_state(batch_size=1, ctx=ctx)
      reconstructed, states = model(image, states)
    #compute difference between reconstructed image and input image 
    reconstructed = reconstructed.numpy()#tf.make_ndarray(reconstructed)
    image_batch1 = image_batch1.numpy()
    diff = np.abs(reconstructed-image_batch1)
    reconstructed=tf.transpose(reconstructed,perm=[0,3,1,2])
    diff=np.transpose(diff,(0,3,1,2))
    #image
    # in case of stacked frames, we need to compute the regularity score per pixel
    if stacked:
       image    = np.sum(image, axis=1, keepdims=True)
       reconstructed = np.sum(reconstructed, axis=1, keepdims=True)
       diff_max = np.max(diff, axis=1)
       diff_min = np.min(diff, axis=1)
       regularity = diff_max - diff_min
       # perform convolution on regularity matrix
       H = signal.convolve2d(regularity[0,:,:], np.ones((4,4)), mode='same')
    else:
      # perform convolution on diff matrix
      H = signal.convolve2d(diff[0,0,:,:], np.ones((4,4)), mode='same')
    
    # if neighboring pixels are anamolous, then mark the current pixel
    x,y = np.where(H > 4)

    # plt input image, reconstructed image and difference between both
    fig, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(10, 5))
    ax0.set_axis_off()
    ax1.set_axis_off()
    ax2.set_axis_off()

    ax0.imshow(image[0,0,:,:], cmap=plt.cm.gray, interpolation='nearest')
    ax0.set_title('input image')
    ax1.imshow(reconstructed[0,0,:,:], cmap=plt.cm.gray, interpolation='nearest')
    ax1.set_title('reconstructed image')
    ax2.imshow(diff[0,0,:,:], cmap=plt.cm.viridis, vmin=0, vmax=1, interpolation='nearest')
    ax2.set_title('diff ')
    ax3.imshow(image[0,0,:,:], cmap=plt.cm.gray, interpolation='nearest')
    ax3.scatter(y,x,color='red',s=0.3)
    ax3.set_title('anomalies')
    plt.axis('off')
    
     # save figure
    counter = counter + 1
    fig.savefig(output_path + "/" + str(counter) + '.png', bbox_inches = 'tight', pad_inches = 0.5)
    plt.close(fig)


# In[2]:


from tensorflow.keras import layers
projection_dim = 64
num_heads = 4
def hms_string(sec_elapsed):
  h = int(sec_elapsed / (60 * 60))
  m = int((sec_elapsed % (60 * 60)) / 60)
  s = sec_elapsed % 60
  return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# In[18]:


def makeEncoder(inputShape, outputUnits):
  
  # Encoder
  encoder = Sequential()

  # Adding Gaussian Noise
  encoder.add(GaussianNoise(stddev = 0, input_shape = inputShape))
  
  # Convolution Block
  encoder.add(Conv2D(filters = 32, kernel_size = (5,5), strides=(2, 2), padding = "same"))
  encoder.add(BatchNormalization())

  encoder.add(LeakyReLU(alpha = 0.3))
  #encoder.add(MaxPool2D(pool_size = (2,2)))

  # Convolution Block
  encoder.add(Conv2D(filters = 64, kernel_size = (5,5), strides=(2, 2), padding = "same"))
  encoder.add(BatchNormalization())
  encoder.add(LeakyReLU(alpha = 0.3))
  #encoder.add(MaxPool2D(pool_size = (2,2)))

  # Convolution Block
  encoder.add(Conv2D(filters = 128, kernel_size = (5,5), strides=(2,2), padding = "same"))
  encoder.add(BatchNormalization())
  encoder.add(LeakyReLU(alpha = 0.3))
  #encoder.add(MaxPool2D(pool_size = (2,2))) 

  encoder.add(Conv2D(filters = 256, kernel_size = (5,5), strides=(2,2), padding = "same"))
  encoder.add(BatchNormalization())
  encoder.add(LeakyReLU(alpha = 0.3))
  encoder.add(Conv2D(filters = 512, kernel_size = (1,1), strides=(2, 2), padding = "same"))
  
  # Latent Space
  encoder.add(Flatten())
  encoder.add(Dense(units = outputUnits))  

  return encoder

def makeDecoder(inputShape):

  # Decoder
  
  decoder = Sequential()
  decoder.add(Input(inputShape))
  decoder.add(Reshape((20,20,16)))
  
  # Convolution Block
  decoder.add(Conv2DTranspose(filters = 128, kernel_size = (5,5), strides=(2, 2), padding = "same"))
  decoder.add(BatchNormalization())
  decoder.add(LeakyReLU(alpha = 0.3))

  # Convolution Block
  decoder.add(Conv2DTranspose(filters = 64, kernel_size = (5,5), strides=(2, 2), padding = "same"))
  decoder.add(BatchNormalization())
  decoder.add(LeakyReLU(alpha = 0.3))

  # Convolution Block
  decoder.add(Conv2DTranspose(filters = 32, kernel_size = (5,5), strides=(2, 2), padding = "same"))
  decoder.add(BatchNormalization())
  decoder.add(LeakyReLU(alpha = 0.3))

  # Convolution Block
  decoder.add(Conv2D(filters = 1, kernel_size = (1,1), strides=(1, 1), padding = "same"))
  #decoder.add(LeakyReLU(alpha = 0.3))
  decoder.add(Activation("tanh"))
  
  return decoder

def makeGenerator(input_dim, latent_dim):

  encoder = makeEncoder(input_dim, latent_dim)
  decoder = makeDecoder((latent_dim,))

  input = Input(input_dim)
  encoderOutput = encoder(input)
  reconstruction = decoder(encoderOutput)

  generator = Model(input, reconstruction)

  return generator


# In[19]:


def makeDiscriminator(inputShape):

  # Discriminator
  discriminator = Sequential()
  
  # Convolution Block
  discriminator.add(Conv2D(filters = 32, kernel_size = (5,5), strides=(2, 2), padding = "same" , input_shape = inputShape))
  discriminator.add(LeakyReLU(alpha = 0.3))

  # Convolution Block
  discriminator.add(Conv2D(filters = 64, kernel_size = (5,5), strides=(2, 2), padding = "same"))
  discriminator.add(BatchNormalization())
  discriminator.add(LeakyReLU(alpha = 0.3))

  # Convolution Block
  discriminator.add(Conv2D(filters = 128, kernel_size = (5,5), strides=(2, 2), padding = "same"))
  discriminator.add(BatchNormalization())
  discriminator.add(LeakyReLU(alpha = 0.3))

  # Convolution Block
  discriminator.add(Conv2D(filters = 256, kernel_size = (5,5), strides=(2, 2), padding = "same"))
  discriminator.add(BatchNormalization())
  discriminator.add(LeakyReLU(alpha = 0.3))
  
  discriminator.add(Flatten())
  discriminator.add(Dense(units = 1, activation = 'sigmoid' ))

  # img = Input(shape = inputShape)
  # validity = discriminator(img)

  return discriminator


# In[32]:


# Change the model path
#modelPath = "/content/drive/MyDrive/PAper2_jou/Models/"
#os.makedirs(modelPath,exist_ok=True)
import os
modelPath="/"
os.makedirs(modelPath,exist_ok=True)
os.chdir(modelPath)


# In[25]:


cross_entropy = tf.keras.losses.BinaryCrossentropy()
# 1 for real images and 0 for fake(generated) ones

def discriminator_loss(realImageOut, fakeImageOut):
    real_loss = cross_entropy(tf.ones_like(realImageOut), realImageOut)
    fake_loss = cross_entropy(tf.zeros_like(fakeImageOut), fakeImageOut)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fakeImageOut): # Ask about this
    return cross_entropy(tf.ones_like(fakeImageOut), fakeImageOut)


# In[26]:


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# In[27]:


input_dim = (160,160,1)
latent_dim = 6400
BATCH_SIZE = 8 # As total number of training images are 310
EPOCHS = 400
#trainingDataset = tf.data.Dataset.from_tensor_slices(trainImages).batch(BATCH_SIZE)


# In[28]:


generator = makeGenerator(input_dim, latent_dim)
discriminator = makeDiscriminator(input_dim)


# In[29]:


checkpoint_prefix = os.path.join(modelPath, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, 
                                generator=generator, discriminator=discriminator)


# In[30]:


# To restore a checkpoint
checkpoint.restore(tf.train.latest_checkpoint(modelPath))


# In[31]:


mse = tf.keras.losses.MeanSquaredError()

def gen_loss1(fake_image,real_image):
  return(mse(fake_image,real_image))


# In[40]:


@tf.function
def train_step(images):
  
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

    generated_images = generator(images, training = True)

    real_output = discriminator(images, training = True)
    fake_output = discriminator(generated_images, training = True)
    gen_loss= gen_loss1(generated_images,images)
    #gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
  return gen_loss, disc_loss


# In[ ]:


train_directory = "/Train/*/*"
test_directory  = "/Test/*/*"


# In[ ]:


files=glob.glob(train_directory)


# In[ ]:


start = time.time()
trainingDataset=create_dataset(train_directory,batch_size=BATCH_SIZE,shuffle=False)
for epoch in tqdm(range(200,300)):
  
  epochStartTime = time.time()

  generatorLossList = []
  discriminatorLossList = []

  for image_batch in trainingDataset:
    image_batch1=tf.transpose(image_batch,perm=[0,2,3,1])
    t = train_step(image_batch1)
    generatorLossList.append(t[0])
    discriminatorLossList.append(t[1])

  g_loss = sum(generatorLossList) / len(generatorLossList)
  d_loss = sum(discriminatorLossList) / len(discriminatorLossList)

  timeElapsed = time.time()-epochStartTime
  print ("Epoch = {}, Generator Loss = {:.7f}, Discriminator Loss = {:.7f}, ".format(epoch+1, g_loss, d_loss) + "Epoch Time : " + hms_string(timeElapsed))

  if((epoch+1)%1==0):  # Save at each 20 epochs
    checkpoint.save(file_prefix = checkpoint_prefix)
    generator.save('generator{:03d}.h5'.format(epoch+1))
    discriminator.save('discriminator{:03d}.h5'.format(epoch+1))

    print("Models saved at {} epochs".format(epoch+1))


elapsed = time.time()-start
print ("Total Training time: " + hms_string(elapsed))


# In[6]:


import tensorflow as tf
generator=tf.keras.models.load_model('')
discriminator=tf.keras.models.load_model('')


# In[ ]:


image_batch.shape,image_batch1.shape
#type(image_batch)


# In[15]:


counter=0


# In[7]:


import scipy
from scipy import signal
#import mxnet as mx

from matplotlib import pyplot as plt
from matplotlib import colors
#from mxnet import gluon,nd
import glob
import numpy as np
import os
from PIL import Image, ImageOps

# create dataloader (batch_size, 1, 100, 100)
def create_dataset(path, file,batch_size, shuffle):
  #files = glob.glob(path)
  files=sorted(file)
  data = np.zeros((len(files),1,160,160))

  for idx, filename in enumerate(files):
    im = Image.open(filename)
    im = ImageOps.grayscale(im)
    im = im.resize((160,160))
    data[idx,0,:,:] = np.array(im, dtype=np.float32)/255.0

  #dataset = gluon.data.ArrayDataset(mx.nd.array(data, dtype=np.float32))
  #dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, last_batch="rollover", shuffle=shuffle)
  dataloader = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)

  return dataloader

# create dataloader (batch_size, 10, 227, 227)
def create_dataset_stacked_images(path, batch_size, shuffle, augment=True):
  #files=sorted(file)
  files = sorted(glob.glob(path))
  if augment:
    files = files + files[2:] + files[4:] + files[6:] + files[8:]
  data = np.zeros((int(len(files)/10),10,160,160))
  i, idx = 0, 0
  for filename in files:
    im = Image.open(filename)
    im = im.resize((160,160))
    data[idx,i,:,:] = np.array(im, dtype=np.float32)/255.0
    i = i + 1
    if i > 9: 
      i = 0
      idx = idx + 1
  #dataset = gluon.data.ArrayDataset(mx.nd.array(data, dtype=np.float32))
  #dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, last_batch="rollover", shuffle=shuffle)
  dataloader = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)

  return dataloader

# perform inference and plot results
def plot_images(counter,path, model, file,output_path="img", stacked=False, lstm=False):

  if not stacked:
    dataloader = create_dataset(path, file,batch_size=1, shuffle=False)
  else:
    dataloader = create_dataset_stacked_images(path, batch_size=1, shuffle=False, augment=False)

  #counter = 0
  #model.load_parameters(params_file, ctx=ctx)
  
  try:
    os.mkdir(output_path,exist_ok=True)
  except:
    pass

  for image in dataloader:
    
    # perform inference
    #image = image.as_in_context(ctx)
    #if not lstm:
      #image_batch1=tf.transpose(image_batch,perm=[0,2,3,1])
      #reconstructed = model(image)
    #else:
      #states = model.temporal_encoder.begin_state(batch_size=1, ctx=ctx)
      #reconstructed, states = model(image, states)
    #compute difference between reconstructed image and input image 
    #reconstructed = reconstructed.asnumpy()
    #image = image.asnumpy()
    #diff = np.abs(reconstructed-image)

    if not lstm:
      image_batch1=tf.transpose(image,perm=[0,2,3,1])
      reconstructed = model(image_batch1)
    else:
      states = model.temporal_encoder.begin_state(batch_size=1, ctx=ctx)
      reconstructed, states = model(image, states)
    #compute difference between reconstructed image and input image 
    reconstructed = reconstructed.numpy()#tf.make_ndarray(reconstructed)
    image_batch1 = image_batch1.numpy()
    diff = np.abs(reconstructed-image_batch1)
    reconstructed=tf.transpose(reconstructed,perm=[0,3,1,2])
    diff=np.transpose(diff,(0,3,1,2))
    #image
    # in case of stacked frames, we need to compute the regularity score per pixel
    if stacked:
       image    = np.sum(image, axis=1, keepdims=True)
       reconstructed = np.sum(reconstructed, axis=1, keepdims=True)
       diff_max = np.max(diff, axis=1)
       diff_min = np.min(diff, axis=1)
       regularity = diff_max - diff_min
       # perform convolution on regularity matrix
       H = signal.convolve2d(regularity[0,:,:], np.ones((4,4)), mode='same')
    else:
      # perform convolution on diff matrix
      H = signal.convolve2d(diff[0,0,:,:], np.ones((4,4)), mode='same')
    
    # if neighboring pixels are anamolous, then mark the current pixel
    x,y = np.where(H > 4)

    # plt input image, reconstructed image and difference between both
    fig, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(10, 5))
    ax0.set_axis_off()
    ax1.set_axis_off()
    ax2.set_axis_off()

    ax0.imshow(image[0,0,:,:], cmap=plt.cm.gray, interpolation='nearest')
    ax0.set_title('input image')
    ax1.imshow(reconstructed[0,0,:,:], cmap=plt.cm.gray, interpolation='nearest')
    ax1.set_title('reconstructed image')
    ax2.imshow(diff[0,0,:,:], cmap=plt.cm.viridis, vmin=0, vmax=1, interpolation='nearest')
    ax2.set_title('diff ')
    ax3.imshow(image[0,0,:,:], cmap=plt.cm.gray, interpolation='nearest')
    ax3.scatter(y,x,color='red',s=0.3)
    ax3.set_title('anomalies')
    plt.axis('off')
    
     # save figure
    counter = counter + 1
    fig.savefig(output_path + "/" + str(counter) + '.png', bbox_inches = 'tight', pad_inches = 0.5)
    plt.close(fig)
  return counter


# In[9]:


files=glob.glob(test_directory)


# In[10]:


len(files)
files=sorted(files)
len(files)


# In[11]:


len(files[0:10])
range(int(len(files)/500) +1)
#len(files[22000:])


# In[ ]:


import tensorflow as tf
generator=tf.keras.models.load_model('generator299.h5')
discriminator=tf.keras.models.load_model('discriminator299.h5')


# In[ ]:


model=generator
output_path='/'
os.makedirs(output_path,exist_ok=True)
plot_images( train_directory, model,output_path)# params_file, ctx, )

