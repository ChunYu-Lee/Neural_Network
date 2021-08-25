# Neural_Network
There are four different models in this repository. Each of them is put under a speciifc folder.  
I build the FeedForward Neural Network, Recurrent Neural Network, Convolutional Neural Network using Java.
Furthermore, under the folder named Framework, I utilize pytorch to build up a CNN model.  

Here the action of how to run the code.  

## FeedForward Neural Network:

You should go to FeedForward folder and try:  
>Run this: java PA13GradientDescent xor small minibatch 10 l2_norm 50 0.1 0.01  

It will run gradient descent on the xor data set with minibatch gradient with a batch size of 10 for 50 epochs, initializing node bias to 0.1, using a learning rate of 0.01, without mu and layers setting.  

java PA13GradientDescent \<data set> \<network type> \<gradient descent type> \<loss function> \<epochs> \<bias> \<learning rate>  

The detail please check FeedForward_Neural_Network/PA13GradientDescent.java.  

>Run this: java PA14GradientDescent mushroom minibatch 20 softmax 100 0.1 0.01 0.9 10 10  

It will run gradient descent on the mushroom data set with minibatch gradient with a batch size of 20 for 100 epochs, initializing node bias to 0.1, using a learning rate of 0.01, a mu of 0.9, and a network with two layers each with 10 nodes.  

java PA14GradientDescent \<data set> \<gradient descent type> \<batch size> \<loss function> \<epochs> \<bias> \<learning rate> \<mu> \<layer_size_1 ... layer_size_n>  

The detail please check FeedForward_Neural_Network/PA14GradientDescent.java.    



## Recurrent Neural Network(RNN):  

You should go to Recurrent Neural Network folder and try:  

>Run this: java GradientDescent flights_small sigmoid jordan kaiming stochastic 2 l2_norm 50 0.0 0.005 0.5 0.05 1.0 10  

It will run gradient descent on the flights_small data set with sigmoid node type, jordan network, stochastic gradient with a batch size of 2 for 50 epochs, initializing node bias to 0.5 and weights through kaiming, using a learning rate of 0.05, a mu of 1.0 with low threshold at 0.05 and high threshold at 1.0, and a network with two layers each with 10 nodes.  

>Run this: java GradientDescent flights_small LSTM jordan kaiming stochastic 2 l2_norm 50 0.0 0.005 0.5 0.05 1.0 10  

It will run gradient descent on the flights_small data set with LSTM node type, jordan network, stochastic gradient with a batch size of 2 for 50 epochs, initializing wieghts and node bias   through kaiming, using a learning rate of 0.005, a mu of 0.5 with low threshold at 0.05 and high threshold at 1.0, and a network with two layers each with 10 nodes.  

java GradientDescent \<data set> \<rnn node type> \<network type> \<initialization type> \<gradient descent type> \<batch size> \<loss function> \<epochs> \<bias> \<learning rate> \<mu> \<low threshold> \<high threshold> \<layer_size_1 ... layer_size_n>  

The detail please check Recurrent_Neural_Network/GradientDescent.java.   



## Convolutional Neural Network(CNN):

You should go to Convolutional Neural Network folder and try:

>Run this: java GradientDescent mnist lenet5 xavier 100 softmax 10 0.5 0.0005 0.5 0 0 0 0 0  

It will run gradient descent on the mnist data set with lenet5 network, minibatch gradient with a batch size of 100 for 10 epochs, initializing node bias to 0.5 and weights through xavier, using a learning rate of 0.0005, a mu of 0.5.

>Run this: java GradientDescent mnist lenet5 kaiming 100 softmax 35 0.5 0.0003 0.5 0 0 0 0 0

It will run gradient descent on the mnist data set with lenet5 network, minibatch gradient with a batch size of 100 for 35 epochs, initializing node bias to 0.5 and weights through kaiming, using a learning rate of 0.0003, a mu of 0.5.

java GradientDescent \<data set> \<network type> \<initialization type> \<batch size> \<loss function> \<epochs> \<bias> \<learning rate> \<mu> \<use dropout> \<input dropout rate> \<hidden dropout rate> \<use batch normalization> \<batch norm alpha>  

The detail please check Convolutional_Neural_Network/GradientDescent.java.  


## Framework(PyTorch)
You should go to Framework folder and try:  

>Run this: Python framework.py  

>Run this: Python framework_add_batchnorm.py  

These two models are build with pytorch. It will run with mnist data set. Both model have 81678 parameters and the second one adds a batchnorm after every ReLU layer.

The detail please check Framework/framework.py. 