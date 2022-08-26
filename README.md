# Receiver-Function-Quality-Control
the code for the paper of Deep Learning for Quality Control of Receiver Functions
DOI 10.3389/feart.2022.921830

**Data**:
/data/x_all.npy: RF data with magnitudes > 5.5 used for training.

/data/y_all.npy: Label of RFs in x_all.npy by manually quality control.

/data/small_earthquake_RF.npy: RF data with magnitudes between 5.0 and 5.5 used for test our trained models.

**Model**:
Trained models are in the model folder.


**quality_control.py** contains the functions used in Deep learning model building.

**quality_control.ipynb** shows a example for building and training four deep learning models in the paper.

