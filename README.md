# IFT6269 Team 7
Adversarial Autoencoders and Moment Matching Autoencoders

## Dependencies
* [PyTorch](https://pytorch.org/)
* [NumPy](https://www.numpy.org/)

## Instalation
  ```
  git clone https://github.com/sbleblanc/pgm_project
  cd pgm_project
  ln -s path/to/states states
  ```
  `ln` creates a symbolic link to where you want to put the state(s) of your model(s). Alternatively, one could use `mkdir`.
  
## Usage
### Training one model
`./train <template path> [<template arg1 name>=<template arg1 value> ...]`

e.g. `./train path/to/my/template.template path=states/mymodel.tar h_num=100 lr=0.01 optimizer=SGD`

Notes: The provided template write a saves a lot. Be conscious of where states is located if you are on a cluster.

## Templates
* Templates are python code with "holes" known as template args
* Template have .template extension
* Template files are located in ./training
* Templates "holes" syntaxe is: `${my_variable_name=default_value}` e.g. `num_hid_units = ${h_num=100}`
