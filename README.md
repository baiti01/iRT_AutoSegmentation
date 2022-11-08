# Introduction
This code is used to train a deep learning-based 3D Auto-segmentation model.

# Usage
1: adapt the configuration file within the folder "experiments"

2: optional: write your visualization code, and put it into the folder "lib/analyze/" with the file name "visualize_{your_model_name}". The default one will be used if the customized visualization code is not provided. Visualization is not mandatory. For the first run, one can turn off the visualization part.

3: customize your dataset within the folder "lib/dataset/" (In this repo, we use "AMOS22.py" as an example)

4: customize your model within the folder "lib/model/" (In this repo, we use "StandardSegmentation.py" as an example)

5: Setup your CUDA device ID and train the model by running:
```console
bash run.sh
```

# Features
- Automatically visualize the testing results and organize the images in an organ-by-organ fashion.
- For reproducibility, we move all the key files to the output folder.
