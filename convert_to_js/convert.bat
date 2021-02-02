@echo off

rem Convert a TensorFlow model to a TensorFlow.js model.
rem Paramters:
rem %1 path to model

tensorflowjs_converter %1 %1js --quantize_float16