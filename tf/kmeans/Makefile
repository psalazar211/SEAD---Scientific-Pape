all:	model.tflite model_edgetpu.tflite

model.tflite:	aggregated.fann
	./tf.py

model_edgetpu.tflite:	model.tflite
	edgetpu_compiler -s model.tflite
