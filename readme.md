# Implementation for "Learning to Reason with Neural Networks: Generalization, Unseen data and Boolean Measures"

This is an implementation for "Learning to Reason with Neural Networks: Generalization, Unseen data and Boolean Measures" using PyTorch. 

Note that in our implantation we always consider l2 loss, however, in the paper we define generalization error equal to half of the l2 loss. 

### Contents:

- `models.py`, `token_transformer.py`, and `mixer.py` contain the code for the MLP, Transformer and MLP-Mixer model respectively. 
- `utilities.py` includes some function definitions and other tools. 
- `neural_net.py` is the main file for the experiments. Some hyperparameters can be sent to it by command line arguments, while some settings can be easily modified in the code. 
- `linear_exp.py` is the code used for experiments on linear neural networks. 
- `sample_scripts.sh` includes some sample commands that can be used to lunch experiments.



