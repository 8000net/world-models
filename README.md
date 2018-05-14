# world-models
Implementation of https://arxiv.org/abs/1803.10122

## Requirments
keras  
tensorflow  
numpy  
pycma  

## Training Process
1. `collect_data.py`
2. `vae.py`
3. `collect_z.py`
4. `rnn.py`
5. `train_env.py`

The controller is currently configured for training with the V model only (only z as input, no h).

Reward over episodes during training, with population of 12, and 4 trials per episode:
![](https://github.com/8000net/world-models/raw/master/fitness.png)


Environment used (removed zoom and indicators):  
https://github.com/justinledford/gym/tree/car-remove-zoom


## Data / Models
Data: https://s2.smu.edu/~jledford/ml/world-models/data.tar.gz  
Models: https://s2.smu.edu/~jledford/ml/world-models/models.tar.gz  


## References
https://worldmodels.github.io/  
https://blog.keras.io/building-autoencoders-in-keras.html  
http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/  
https://medium.com/applied-data-science/how-to-build-your-own-world-model-using-python-and-keras-64fb388ba459  
