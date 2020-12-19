# Model Training

## Files

- `ResNet.py` trains a ResNet-101 model on the dataset.
-  `DenseNet.py` trains a DenseNet-121 model on the dataset.
-  `dataset.py` creates a new train/test dataset by preprocessing the raw dataset.
-  `utils.py` helps do some of the hardcore image processing in dataset.py.
-  `averagemeter.py` helps keep track of a bunch of averages when training.
-  `leafsnap-dataset-images.csv` is the CSV file corresponding to the dataset.
-  `requirements.txt` contains the pip requirements to run the code.



## Dataset

The dataset used to train our model is [Leafsnap Dataset](https://www.kaggle.com/xhlulu/leafsnap-dataset) which is 890MB.



## Training Model

To train the ResNet-101 model, run `python ResNet.py`

To train the DenseNet-121 model, run `python DenseNet.py`





