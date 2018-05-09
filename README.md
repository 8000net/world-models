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
TODO: upload data and models
