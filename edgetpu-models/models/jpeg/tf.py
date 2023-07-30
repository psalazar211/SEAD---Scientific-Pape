#!/usr/bin/python3

import tensorflow as tf

model = tf.keras.models.Sequential([
	tf.keras.layers.Input(dtype=float,shape=(64,)),
	tf.keras.layers.Dense(16, activation='sigmoid'),
	tf.keras.layers.Dense(64)
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
	inFloat=[]
	for j in range(len(inStr)):
		inFloat.append(float(inStr[j]))
	inTrain.append(inFloat)
	outStr = outLine.split()
	outFloat=[]
	for j in range(len(outStr)):
		outFloat.append(float(outStr[j]))
	outTrain.append(outFloat)

for i in range(numTest):
	inLine = f.readline()
	outLine = f.readline()
	if not inLine or not outLine:
		break
	inFloat = []
	for j in range(len(inStr)):
		inFloat.append(float(inStr[j]))
	inTest.append(inFloat)
	outStr = outLine.split()
	outFloat=[]
	for j in range(len(outStr)):
		outFloat.append(float(outStr[j]))
	outTest.append(outFloat)

f.close()

model.fit(inTrain, outTrain, epochs=1000)
model.evaluate(inTest, outTest, verbose=1)

for i in range(numTest):
	test=[]
	test.append(inTest[i])
	#print(model.predict(test),outTest[i])

model.save('model')

def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((inTrain)).batch(1).take(100):
    yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8
#converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
