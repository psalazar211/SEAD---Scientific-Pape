#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#define EDGETPU

#ifdef EDGETPU
#include "edgetpu.h"
#endif

#include <iostream>

static tflite::StderrReporter error_reporter;

static std::unique_ptr<tflite::FlatBufferModel> model;
static std::unique_ptr<tflite::Interpreter> model_interpreter;
static tflite::ops::builtin::BuiltinOpResolver resolver;

#ifdef EDGETPU
static std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;

static std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* edgetpu_context) {

  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

  //std::cout << "Building interpreter" << std::endl;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
  }

  //std::cout << "Binding interpreter" << std::endl;
  // Bind context with interpreter
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }

  return interpreter;
}
#endif

static std::unique_ptr<tflite::Interpreter> BuildInterpreter(const tflite::FlatBufferModel& model) {
  //std::cout << "Building interpreter" << std::endl;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
  }

  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }

  return interpreter;
}

void tfInit(const char* filename) {
  // Load model
  model = tflite::FlatBufferModel::BuildFromFile(filename, &error_reporter);
  if (!model) std::cerr << "BuildFromFile failed\n" << std::endl;

#ifdef EDGETPU
  // Open TPU
  edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!edgetpu_context) {
    std::cerr << "OpenDevice failed\n" << std::endl;
  }

  // Build interpreter
  model_interpreter = BuildEdgeTpuInterpreter(*model, edgetpu_context.get());
  if (!model_interpreter) {
    std::cerr << "BuildEdgeTpuInterpreter failed\n" << std::endl;
  }

#else
  // Build interpreter
  model_interpreter = BuildInterpreter(*model);
  if (!model_interpreter) {
    std::cerr << "BuildInterpreter failed\n" << std::endl;
  }
#endif
}

void tfClose(void) {
  model_interpreter.reset();
  edgetpu_context.reset();
}

void tfInfer(int numin, float* inPtr, int numout, float* outPtr) {
  TfLiteTensor *input = nullptr;

  input = model_interpreter->input_tensor(0);

  for (int i=0; i<numin; i++) {
    input->data.f[i] = inPtr[i];
    //std::cout << inPtr[i] << "\t";
  }

  try {
     model_interpreter->Invoke();
  } catch (...) {
    std::cerr << "Invoke failed\n" << std::endl;
  }

  //std::cout << "->\t";

  TfLiteTensor *output = nullptr;
  output = model_interpreter->output_tensor(0);

  for (int i=0; i<numout; i++) {
    outPtr[i] = output->data.f[i];
    //std::cout << outPtr[i] << "\t";
  }
  //std::cout << std::endl;
}
