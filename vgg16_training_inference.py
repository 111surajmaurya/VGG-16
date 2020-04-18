from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Flatten
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

img_width = 224
img_height = 224
num_channels = 3
batch_size = 32

base_model = VGG16(input_shape = (img_height,img_width,num_channels),weights='imagenet', include_top=False)

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print(model.summary())

l = len(base_model.layers)
for i in range(l):
  if(i < int(0.7*l)):
    base_model.layers[i].trainable = False
  else:
    base_model.layers[i].trainable = True


#for layer in model.layers:
#  print(layer.name, layer.trainable)

sgd =SGD(lr=0.001, decay=1e-6, momentum=0.9)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics =['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/content/data/train',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
        '/content/data/validation',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

#print(train_generator.samples, validation_generator.samples)
#print(train_generator.filenames)
#print(validation_generator.labels)

#change val_acc to val_accuracy in case of error

filepath="vgg16_-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')

epochs = 25

model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples//batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples//batch_size,
        callbacks= [checkpoint])



#uncomment for inference
"""
from keras.models import load_model
from PIL import Image
import cv2
import matplotlib.pyplot as plt 
import numpy as np

classes = ['cat', 'dog']

model_path = "/path/of/weight/file"

model = load_model(model_path)

file = "/path/of/image"

img_obj = Image.open(file)
img_org = np.array(img_obj)
img = cv2.resize(img_org,(224,224))
img = np.expand_dims(img, axis=0)/255.0

prob = model.predict(img)

id = 0
if(prob>=0.5):
  id = 1
print(classes[id])
#plt.imshow(img_org)
#plt.show()

"""






