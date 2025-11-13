# Confidence Landscape App
## Overview
- The [Confidence Landscape App](https://confidencelandscape.streamlit.app/) allows users to select perturbations to apply to an image, and see how it changes an image classification model's prediction confidence.
- The app uses GradCAM to create a visual heatmap to improve interpretability.

## Model Details
- **Image Classification Models**:
  - [ResNet18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
  - [MobileNet_V3_Large](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html)
  - [EfficientNet_B0](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html)

## Code Description
- [Data\images](https://github.com/rrdu/ConfidenceLandscape/tree/main/data/images): sample images, 5 per category
  - [airplane](https://github.com/rrdu/ConfidenceLandscape/tree/main/data/images/airplane)
  - [cat](https://github.com/rrdu/ConfidenceLandscape/tree/main/data/images/cat)
  - [dalmatian](https://github.com/rrdu/ConfidenceLandscape/tree/main/data/images/dalmatian)
  - [jaguar](https://github.com/rrdu/ConfidenceLandscape/tree/main/data/images/jaguar)
  - [school_bus](https://github.com/rrdu/ConfidenceLandscape/tree/main/data/images/school_bus)
  - [run_button_pic.png](https://github.com/rrdu/ConfidenceLandscape/blob/main/data/images/run_button_pic.png)
- [models](https://github.com/rrdu/ConfidenceLandscape/tree/main/models): code relevant to loading models
  - [model_loader.py](https://github.com/rrdu/ConfidenceLandscape/blob/main/models/model_loader.py): loads the image classification models
  - [model_map.py](https://github.com/rrdu/ConfidenceLandscape/blob/main/models/model_map.py): maps the image classification model names to the appropriate loaders
- [utils](https://github.com/rrdu/ConfidenceLandscape/tree/main/utils): useful functions for the main app interface to use
  - [cache.py](https://github.com/rrdu/ConfidenceLandscape/blob/main/utils/cache.py): creates and loads a cache to avoid making a new grid every time 
  - [gradcam.py](https://github.com/rrdu/ConfidenceLandscape/blob/main/utils/gradcam.py): runs the GradCAM function
  - [heatmap_overlay.py](https://github.com/rrdu/ConfidenceLandscape/blob/main/utils/heatmap_overlay.py): creates the heatmap overlay filter for GradCAM
  - [landscape.py](https://github.com/rrdu/ConfidenceLandscape/blob/main/utils/landscape.py): creates the confidence landscape and runs the perturbations
  - [perturbations.py](https://github.com/rrdu/ConfidenceLandscape/blob/main/utils/perturbations.py): applies the chosen perturbations (brightness, rotation, noise) to an image
  - [plotting.py](https://github.com/rrdu/ConfidenceLandscape/blob/main/utils/plotting.py): plots the interactable plotting landscape using Plotly
- [.gitignore](https://github.com/rrdu/ConfidenceLandscape/blob/main/.gitignore): lists assets for Streamlit and Github to ignore when running/updating code
- [confidence_landscape_app.py](https://github.com/rrdu/ConfidenceLandscape/blob/main/confidence_landscape_app.py): main script to run the Streamlit app interface
- [config.py](https://github.com/rrdu/ConfidenceLandscape/blob/main/config.py): lists the relevant model classes, image folder names, axes of perturbation
- [requirements.txt](https://github.com/rrdu/ConfidenceLandscape/blob/main/requirements.txt): necessary requirements to install to run the app locally
- [runtime.txt](https://github.com/rrdu/ConfidenceLandscape/blob/main/runtime.txt): lists the necessary Python version to run the app
- [venv_instructions.txt](https://github.com/rrdu/ConfidenceLandscape/blob/main/venv_instructions.txt): instructions for creating a virtual environment to run the app locally
