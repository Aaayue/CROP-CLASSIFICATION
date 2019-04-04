# CROP_CLASSIFICATION
------

This repo is all about the models we used for crop classification.

Usually LSTM is the prefered choice for crop classification based on time series date. while since the LSTM model is time-costing, some other models are also used in project, such as MLP, SVM, DT, all the model parameters can insert from command lines. 

Please use "-h" for help.

For LSTM, the input file should be a dictionary which contains time and source as keys, but for other model, the input should be np.ndarray, the programms read the input file and directly throw it to model for training.

<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a>
