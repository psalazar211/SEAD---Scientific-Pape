#ifndef __TF_H__
#define __TF_H__

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

void tfInit(const char* filename);
void tfClose(void);
void tfInfer(int numin, float* inPtr, int numout, float* outPtr);

#endif
