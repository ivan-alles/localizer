# Convert a model to TensorFlow.js
Conversion to TensorFlow.js requires a separate pipenv environment, otherwise TF will stop using GPU.
I assume that you have done the previous steps and have yarn installed.

1. In a new terminal window, go to `convert_to_js` directory.
2. Install python dependencies: `pipenv sync`.
3. Convert the model to TensorFlow.js format: `pipenv run convert.bat PATH_TO_MODEL`.
