# Canary CREST Research Code

Source code regarding my research during the Canary CREST Research Program, Summer 2021. 
Project: "Registration of the T2 and DWI b1200 MRI
Sequences of the Prostate". 

## Structure

This repository is structured as follows (see [here](https://rosikand.github.io/research/canary.html) for more details on the research): 

- `classical`: contains the source code that uses SimpleITK for the classical registration methods including linear (`linear_methods.py`) and deformable registration methods (`deformable_methods.py`). 

- `learning-based`: contains the source code (that uses PyTorch as well as Tensorflow) for the the learning-based methods used in the experiment. Includes both the generative approach (learning the moved image directly using a neural network as described in section 2.3, stored in `direct_model_tf.py`) and the deformable learning-based approach (as described in section 2.3.1, stored in `deformable_model.py`). A generative autoencodeer PyTorch model not mentioned in the report is included in `direct_model_torch.py`. 


