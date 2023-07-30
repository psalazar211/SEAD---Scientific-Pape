#!/usr/bin/python3

import tensorflow as tf

model = tf.keras.models.Sequential([
	tf.keras.layers.Input(dtype=float,shape=(1,)),
	tf.keras.layers.Dense(4, activation='sigmoid'),
	tf.keras.layers.Dense(4, activation='sigmoid'),
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
