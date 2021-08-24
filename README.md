# Neural_Network
There are four different models in this repository. Each of them is put under a speciifc folder.  
I build the FeedForward Neural Network, Recurrent Neural Network, Convolutional Neural Network using Java.
Furthermore, under the folder named Framework, I utilize pytorch to build up a CNN model.  

Here is the result and action of how to run the code.  

FeedForward Neural Network:

You should go to FeedForward folder and try:  
java PA13GradientDescent xor small minibatch 10 l2_norm 50 0.1 0.01  

It will run gradient descent on the xor data set with minibatch gradient with a batch size of 10 for 50 epochs, initializing node bias to 0.1, using a learning rate of 0.01, without mu and layers setting.  

java PA14GradientDescent mushroom minibatch 20 softmax 100 0.1 0.01 0.9 10 10  

It will run gradient descent on the mushroom data set with minibatch gradient with a batch size of 20 for 100 epochs, initializing node bias to 0.1, using a learning rate of 0.01, a mu of 0.9, and a network with two layers each with 10 nodes.  

Recurrent Neural Network(RNN):  

You should go to Recurrent Neural Network folder and try:  

