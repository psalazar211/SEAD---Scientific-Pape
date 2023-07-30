#!/usr/bin/python3

import tensorflow as tf

model = tf.keras.models.Sequential([
	tf.keras.layers.Input(dtype=float,shape=(1,)),
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
	inTrain.append(float(inLine))
	outStr = outLine.split()
	outFloat=[]
	outFloat.append(float(outStr[0]))
	outFloat.append(float(outStr[1]))
	outTrain.append(outFloat)

for i in range(numTest):
	inLine = f.readline()
	outLine = f.readline()
	if not inLine or not outLine:
		break
	inTest.append(float(inLine))
	outStr = outLine.split()
	outFloat=[]
	outFloat.append(float(outStr[0]))
	outFloat.append(float(outStr[1]))
	outTest.append(outFloat)

f.close()

model.fit(inTrain, outTrain, epochs=100)
model.evaluate(inTest, outTest, verbose=1)

for i in range(numTest):
	test=[]
	test.append(inTest[i])
	print(model.predict(test),outTest[i])

model.save('model')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
