import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from PIL import Image


#!!matrix array different, matrix is always 2D!!

cleanpath='c:/data/images/clean/'
dewpath='c:/data/images/dew/'
frostpath='c:/data/images/frost/'

clean_array=np.empty((0,300,300),dtype=np.uint8)
dew_array=np.empty((0,300,300),dtype=np.uint8)
frost_array=np.empty((0,300,300),dtype=np.uint8)

for i in range(1,201):
    img_clean=Image.open(cleanpath+str(i)+'.bmp')
    data=img_clean.getdata()
    array=np.array(data,dtype=np.uint8)
    array_=np.reshape(array,(1,300,300))
    clean_array=np.row_stack((clean_array,array_))
    print("Now, clean_array's length is :"+str(len(clean_array))+"\n")

for i in range(1,201):
    img_dew=Image.open(dewpath+str(i)+'.bmp')
    data=img_dew.getdata()
    array=np.array(data,dtype=np.uint8)
    array_=np.reshape(array,(1,300,300))
    dew_array=np.row_stack((dew_array,array_))
    print("Now, dew_array's length is :"+str(len(dew_array))+"\n")

for i in range(1,201):
    img_frost=Image.open(frostpath+str(i)+'.bmp')
    data=img_frost.getdata()
    array=np.array(data,dtype=np.uint8)
    array_=np.reshape(array,(1,300,300))
    frost_array=np.row_stack((frost_array,array_))
    print("Now, frost_array's length is :"+str(len(frost_array))+"\n")

print("Now, all training data is already.\n ")

clean_label=np.empty((0,1),dtype=np.int8)
for i in range(0,200):
    clean_label=np.row_stack((clean_label,np.array([0])))

dew_label=np.empty((0,1),dtype=np.int8)
for i in range(0,200):
    dew_label=np.row_stack((dew_label,np.array([1])))

frost_label=np.empty((0,1),dtype=np.int8)
for i in range(0,200):
    frost_label=np.row_stack((frost_label,np.array([2])))

print("Now, all label data is already.\n")
print("Trying to connect all data...\n")

train_data1=np.row_stack((clean_array[0:160],dew_array[0:160],frost_array[0:160])).reshape(480,300,300,1) #80% 2400
test_data1=np.row_stack((clean_array[160:],dew_array[160:],frost_array[160:])).reshape(120,300,300,1) #20% 600

train_label1=np.row_stack((clean_label[0:160],dew_label[0:160],frost_label[0:160]))
test_label1=np.row_stack((clean_label[160:],dew_label[160:],frost_label[160:]))


train_label=keras.utils.to_categorical(train_label1,3)
test_label=keras.utils.to_categorical(test_label1,3)

train_data=train_data1.astype('float32')
test_data=test_data1.astype('float32')
train_data/=255
test_data/=255



print("Training and testing data and label are all already.\n")
print("Now start build CNN.")

model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',border_mode='same',input_shape=(300,300,1)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))

opt=keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.fit(train_data,train_label,batch_size=32,epochs=20,validation_data=(test_data,test_label),shuffle=True) # start training.

score=model.evaluate(test_data,test_label,batch_size=32)
print(score)

'''
model_json=model.to_json() #json file
with open("nice_model.json","w") as json_file:
    json_file.write(model_json)

model.save_weights("nice_model.h5") #HDF5 file

resault=model.predict(train_data[179].reshape(1,300,300,1),batch_size=20,verbose=1)
print(resault)
'''
