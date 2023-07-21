# -*- coding: utf-8 -*-
"""jmeint.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aCZEl8B1WpyYmGuBC9YvHUAAazm0NIDl

Model
"""

import tensorflow as tf

model = tf.keras.models.Sequential([
	tf.keras.layers.Input(dtype=float,shape=(18,)),
	tf.keras.layers.Dense(32, activation='sigmoid'),
	tf.keras.layers.Dropout(0.01),
	tf.keras.layers.Dense(32, activation='sigmoid'),
	tf.keras.layers.Dense(2)
])

model.compile(optimizer='RMSprop', loss=tf.keras.metrics.mean_squared_error, metrics=["mae"])

model.summary()

inTrain = []
outTrain = []
inTest = []
outTest = []

f = open("aggregated.fann", "r")
shape = f.readline()
num=int(shape.split()[0])
numTrain = int(num*0.9)
numTest = int(num*0.1)

for i in range(numTrain):
  inLine = f.readline()
  outLine = f.readline()
  if not inLine or not outLine:
    break
  inStr = inLine.split()
  inFloat = []
  inFloat.append(float(inStr[0]))
  inFloat.append(float(inStr[1]))
  inFloat.append(float(inStr[2]))
  inFloat.append(float(inStr[3]))
  inFloat.append(float(inStr[4]))
  inFloat.append(float(inStr[5]))
  inFloat.append(float(inStr[6]))
  inFloat.append(float(inStr[7]))
  inFloat.append(float(inStr[8]))
  inFloat.append(float(inStr[9]))
  inFloat.append(float(inStr[10]))
  inFloat.append(float(inStr[11]))
  inFloat.append(float(inStr[12]))
  inFloat.append(float(inStr[13]))
  inFloat.append(float(inStr[14]))
  inFloat.append(float(inStr[15]))
  inFloat.append(float(inStr[16]))
  inFloat.append(float(inStr[17]))
  inTrain.append(inFloat)
  outFloat = []
  outFloat.append(float(inStr[0]))
  outFloat.append(float(inStr[1]))
  outTrain.append(outFloat)

for i in range(numTest):
  inLine = f.readline()
  outLine = f.readline()
  if not inLine or not outLine:
    break
  inStr = inLine.split()
  inFloat = []
  inFloat.append(float(inStr[0]))
  inFloat.append(float(inStr[1]))
  inFloat.append(float(inStr[2]))
  inFloat.append(float(inStr[3]))
  inFloat.append(float(inStr[4]))
  inFloat.append(float(inStr[5]))
  inFloat.append(float(inStr[6]))
  inFloat.append(float(inStr[7]))
  inFloat.append(float(inStr[8]))
  inFloat.append(float(inStr[9]))
  inFloat.append(float(inStr[10]))
  inFloat.append(float(inStr[11]))
  inFloat.append(float(inStr[12]))
  inFloat.append(float(inStr[13]))
  inFloat.append(float(inStr[14]))
  inFloat.append(float(inStr[15]))
  inFloat.append(float(inStr[16]))
  inFloat.append(float(inStr[17]))
  inTest.append(inFloat)
  outFloat = []
  outFloat.append(float(inStr[0]))
  outFloat.append(float(inStr[1]))
  outTest.append(outFloat)
f.close()

"""Train model"""

model.fit(inTrain, outTrain, epochs=100)

"""Evaluate"""

ev = model.evaluate(inTest, outTest, verbose=1)

print("MAE", ev[1])