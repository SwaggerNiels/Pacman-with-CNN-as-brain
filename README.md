# Pacman-with-CNN-as-brain

In this simulation pacman uses his vision range in front of him to asses the situation, through the usage
of a convolutional neural network. This distance of an object to pacman is translated into a depth (image resize) and the whole image is overlayed with noise, which can be set to a desired strength.

## Modes
There are 4 modes:

### Pause

The game starts in pause mode, which can be accesed any time by pressing "p"-key within another mode.

### Data collection

This should be used to train the CNN. First activation of "d"-key will start the collection of images. After this "d"-key can be used to add new epochs (train the network with another random configuration of the training data). The accuracy relative to the complete trainingdata set is shown (100% recommended for pacman not to get stuck).

### Field / Stagin

Here is the actual field of pacman. Use "f" to activate this mode, but if CNN is not well trained, pacman can get stuck...
You can also copy the weights and biases from a previously trained network (100% accuracy) by pressing "i"-key (input network). Or use the "o"-key to overwrite this with your current NN weights and biases.

### Lab

Here pacman is fixed in place and is presented with the three objects: "wall", "ghost" and "coin". His responses can be observed stably. 

## Keys to control interface

### up arrow
move pacman up
### down arrow
move pacman down
### left arrow
move pacman left
### right arrow
move pacman right


### q
increase noise in pacman's vision
### a
decrease noise in pacman's vision
### w
increase the frame delay -> lower game speed
### s
decrease the frame delay -> higher game speed
### d
(first) activate data collection mode, (next) add training iteration/epoch to update CNN, accuracy echo'd in terminal
### r
reset the weights and biases in the CNN to random new numbers
### i
load NN preset weights and biases (100% accuracy)
### o
overwrite the NN preset weights and biases with your currenct NN weights and biases
### l
activate the lab game mode
### f
activate the field/staging game mode
### m
set building mode (activate this before loading wall preset!)
### 1-9
either one of the numbers can be used to load a wall present (when in building mode!^)

either one of the numbers can be used to store a wall preset (when not in building mode, carefull for overwrites!)
### p
pause the current game mode
### mouse left click
add wall
### right mouse click
remove wall
### middle mouse click
identify the object under mouse, shown in terminal

