# Neural_Network
There are four different models in this repository. Each of them is put under a speciifc folder.  
I build the FeedForward Neural Network, Recurrent Neural Network, Convolutional Neural Network using Java.
Furthermore, under the folder named Framework, I utilize pytorch to build up a CNN model.  

Here the action of how to run the code.  

FeedForward Neural Network:

You should go to FeedForward folder and try:  
Run this: java PA13GradientDescent xor small minibatch 10 l2_norm 50 0.1 0.01  

It will run gradient descent on the xor data set with minibatch gradient with a batch size of 10 for 50 epochs, initializing node bias to 0.1, using a learning rate of 0.01, without mu and layers setting.  

java PA13GradientDescent <data set> <network type> <gradient descent type> <loss function> <epochs> <bias> <learning rate>  

Where:  

data set can be: 'and', 'or' or 'xor'  

network type can be: 'tiny', 'small' or 'large'  

gradient descent type can be: 'stochastic', 'minibatch' or 'batch'  

loss function can be: 'l1_norm', or 'l2 norm'  

epochs is an integer > 0  

bias is a double  

learning rate is a double usually small and > 0, you could start with 0.3  
--------------------------------------------------------

Run this: java PA14GradientDescent mushroom minibatch 20 softmax 100 0.1 0.01 0.9 10 10  

It will run gradient descent on the mushroom data set with minibatch gradient with a batch size of 20 for 100 epochs, initializing node bias to 0.1, using a learning rate of 0.01, a mu of 0.9, and a network with two layers each with 10 nodes.  

java PA13GradientDescent <data set> <gradient descent type> <batch size> <loss function> <epochs> <bias> <learning rate> <mu> <layer_size_1 ... layer_size_n>  

Where:  

data set can be: 'and', 'or', 'xor', 'iris' or 'mushroom'  

gradient descent type can be: 'stochastic', 'minibatch' or 'batch'  

batch size should be > 0. Will be ignored for stochastic or batch gradient descent.  

loss function can be: 'l1_norm', 'l2 norm', 'svm' or 'softmax'. note that you need to use l1 or l2 norm for the or, and or xor data, and svm or softmax for the iris or mushroom data.  

epochs is an integer > 0  

bias is a double, start with 0.1  

learning rate is a double usually small and > 0, I suggest starting with 0.3  

mu is a double < 1 and typical values are 0.5, 0.9, 0.95 and 0.99  

After this you can specify the layer sizes.  



Recurrent Neural Network(RNN):  

You should go to Recurrent Neural Network folder and try:  

java GradientDescent flights_small sigmoid jordan kaiming stochastic 2 l2_norm 50 0.0 0.005 0.5 0.05 1.0 10

It will run gradient descent on the flights_small data set with stochastic gradient with a batch size of 2 for 50 epochs, initializing node bias to 0.1, using a learning rate of 0.01, a mu of 0.9, and a network with two layers each with 10 nodes.  

java GradientDescent flights_small LSTM jordan kaiming stochastic 2 l2_norm 50 0.0 0.005 0.5 0.05 1.0 10  


java GradientDescent <data set> <rnn node type> <network type> <initialization type> <gradient descent type> <batch size> <loss function> <epochs> <bias> <learning rate> <mu> <low threshold> <high threshold> <layer_size_1 ... layer_size_n>  

Where:  

data set can be: 'penn_small', 'penn_full' or 'flights_small', 'flights_full'  

rnn node type can be: 'linear', 'sigmoid', 'tanh', 'lstm', 'gru', 'ugrnn', 'mgu' or 'delta'  

network type can be: 'feed_forward', 'jordan' or 'elman'  

initialization type can be: 'xavier' or 'kaiming'  

gradient descent type can be: 'stochastic', 'minibatch' or 'batch'  

batch size should be > 0. Will be ignored for stochastic or batch gradient descent
loss function can be: 'l1_norm', 'l2_norm', 'svm' or 'softmax'  

epochs is an integer > 0  

bias is a double  

learning rate is a double usually small and > 0  

mu is a double < 1 and typical values are 0.5, 0.9, 0.95 and 0.99  

low threshold is a double value to use as the threshold for gradient boosting (0.05 recommended), if it is < 0, gradient boosting will not be used  

high threshold is a double value to use as the threshold for gradient scaling (1.0 recommended), if it is < 0, gradient scaling will not be used  

layer_size_1..n is a list of integers which are the number of nodes in each hidden layer  



Convolutional Neural Network(CNN):

  