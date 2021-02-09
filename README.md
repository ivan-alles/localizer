![localizer](https://github.com/ivan-alles/localizer/workflows/CI/badge.svg)

# Localizer

Localizer is a neural network predicting object positions and orientation on 2d images.

Check out a [browser app](https://ivan-alles.github.io/localizer/) finding your hands on a live camera video.

## Setup
I tested these instruction under Windows. 

1. To run neural networks on a GPU (highly recommended) 
   install (if not done yet) the required drivers, etc. for **[TensorFlow 2](https://www.tensorflow.org/install/gpu)**.
2. Get the source code into your working folder.
3. Install the dependencies with `pipenv sync`.
4. Activate the pipenv environment with `pipenv shell`.
5. Run `set PYTHONPATH=.`.  

## Run the hands-on python app

You can interactively train and run a model on images from your web camera in the hands-on demo app. Run 
`python localizer\hands_on_demo.py` and follow the on-screen instructions.

## Converting an existing dataset

A dataset required by the localizer is a number of images annotated in a simple JSON file. This is a list of images, each containing a list of objects 
with position, orientation and category:

```json
[
 {
  "image": "myimage.png",
  "objects": [
   {
    "category": 0,
    "origin": {
     "x": 100,
     "y": 200,
     "angle": 3.14159
    }
   }
  ]
 }
]
```

You need to convert your dataset to this format to use it with the localizer.

## Creating a new dataset 

You can use [Anno](https://github.com/urobots-io/anno/) to manually label images. To do this:

1. Download and install [Anno](https://github.com/urobots-io/anno/).
2. Copy `localizer\dataset_template.anno` into the directory with your images and rename it (e.g. `mydataset.anno`).
3. Open `mydataset.anno` with Anno.
4. Label objects with the **object** marker. 
5. Label images without any objects with the **empty** marker.

This anno file can be specified in the model configuration:

```json
{
  "dataset": "path/mydataset.anno"
} 
```

## Training a model
Run `python localizer\train.py PATH_TO_MODEL_CONFIG`. 

For example: `python localizer\train.py models\tools\config.json`.

This will train and save a Tensorflow model under `models\tools\model.tf`.

## Running a model
Run `python localizer\predict_for_images.py PATH_TO_MODEL_CONFIG IMAGES_DIR`.
 
For example: `python localizer\predict_for_images.py models\tools\config.json datasets\tools`.

## Use transfer learning
Transfer learning reduces the number of labeled images required for training by more than 10 times:

<img src="./assets/Training performance.svg">

To use transfer learning in training, add the following to the configuration:

```json
{
  "transfer_learning_base": "models/transfer_learning_base/features.tf",
  "pad_to": 32
} 
```
The input shape must be divisible by 32, for example:

```json
{
  "input_shape": [
    384,
    384,
    3
  ]
} 
```