"""
File: direct_model_tf.py  
-------------------
This file contains the Tensorflow source code for the direct, generative 
learning-based registration model as described in section 2.3. 
""" 

import numpy as np
import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt 

INPUT_DIM = 7864320
OUTPUT_DIM = 6291456 


# Model 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(INPUT_DIM,)),
    tf.keras.layers.Dense(28, activation='relu'),
    tf.keras.layers.Dense(OUTPUT_DIM, activation='sigmoid')
])

# Training 
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(processed_train, fixed_train,
                epochs=250, 
                batch_size=1,
                shuffle=True)

# Testing  
def dice_coef(y_true, y_pred, smooth=1):
  # adapted from 
  # https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
  intersection = K.sum(y_true * y_pred)
  union = K.sum(y_true) + K.sum(y_pred)
  dice = K.mean((2. * intersection + smooth)/(union + smooth))
  return dice

predicted_imgs_batch = model.predict(processed_test_set) 

results_total = 0
for i in range(len(predicted_imgs_batch)):
  result = dice_coef(fixed_test_batch[i], predicted_imgs_batch[i])
  results_total += float(result)
print("The average Dice score on the test set is: ", results_total/len(predicted_imgs_batch))

