package network;

import java.util.List;

import data.Image;
import data.ImageDataSet;

import util.Log;


public class ConvolutionalNeuralNetwork {
    //this is the loss function for the output of the neural network
    LossFunction lossFunction;

    //this is the total number of weights in the neural network
    int numberWeights;

    //specifies if the CNN will use dropout
    boolean useDropout;
    //the dropout for nodes in the input layer
    double inputDropoutRate;
    //the dropout for nodes in the hidden layer
    double hiddenDropoutRate;

    //specify if the CNN will use batch normalization
    boolean useBatchNormalization;

    //the alpha value used to calculate the running
    //averages for batch normalization
    double alpha;

    //layers contains all the nodes in the neural network
    ConvolutionalNode[][] layers;

    public void createSmallNoPool(ActivationType activationType, int batchSize, int inputChannels, int inputY, int inputX, int inputPadding, int outputLayerSize) throws NeuralNetworkException {
        layers = new ConvolutionalNode[5][];

        layers[0] = new ConvolutionalNode[1];
        ConvolutionalNode inputNode = new ConvolutionalNode(0, 0, NodeType.INPUT, activationType /*doesn't matter for input*/, inputPadding /*padding*/ , batchSize, inputChannels, inputY, inputX, useDropout, inputDropoutRate, false, 0.0);
        layers[0][0] = inputNode;


        //first hidden layer has 4 nodes
        layers[1] = new ConvolutionalNode[4];
        for (int i = 0; i < 4; i++) {
            ConvolutionalNode node = new ConvolutionalNode(1, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 20, 20, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[1][i] = node;

            numberWeights += node.getNumberWeights();

            new ConvolutionalEdge(layers[0][0], node, inputChannels /* will be 1 for MNIST or 3 for CIFAR-10 to get the other feature maps to only have 1 channel*/, 13, 13);

            numberWeights += inputChannels * 13 * 13;
        }


        //second hidden layer also has 4 nodes
        layers[2] = new ConvolutionalNode[4];
        for (int i = 0; i < 4; i++) {
            ConvolutionalNode node = new ConvolutionalNode(2, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 10, 10, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[2][i] = node;

            numberWeights += node.getNumberWeights();

            for (int j = 0; j < 4; j++) {
                new ConvolutionalEdge(layers[1][j], node, 1, 11, 11); //11x11 to get down to 10x10

                numberWeights += 11 * 11;
            }
        }


        //third hidden layer is dense with 10 nodes
        layers[3] = new ConvolutionalNode[10];
        for (int i = 0; i < 10; i++) {
            ConvolutionalNode node = new ConvolutionalNode(3, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[3][i] = node;

            numberWeights += node.getNumberWeights();

            for (int j = 0; j < 4; j++) {
                new ConvolutionalEdge(layers[2][j], node, 1, 10, 10);

                numberWeights += 10 * 10;
            }
        }

        //output layer is dense with 10 nodes
        layers[4] = new ConvolutionalNode[10];
        for (int i = 0; i < 10; i++) {
            //no dropout or BN on output nodes
            ConvolutionalNode node = new ConvolutionalNode(4, i, NodeType.OUTPUT, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, false, 0.0, false, 0.0);
            layers[4][i] = node;

            //no bias on output nodes

            for (int j = 0; j < 10; j++) {
                new ConvolutionalEdge(layers[3][j], node, 1, 1, 1);

                numberWeights += 1;
            }
        }
    }

    public void createSmall(ActivationType activationType, int batchSize, int inputChannels, int inputY, int inputX, int inputPadding, int outputLayerSize) throws NeuralNetworkException {
        layers = new ConvolutionalNode[5][];

        layers[0] = new ConvolutionalNode[1];
        ConvolutionalNode inputNode = new ConvolutionalNode(0, 0, NodeType.INPUT, activationType /*doesn't matter for input*/, inputPadding /*padding*/ , batchSize, inputChannels, inputY, inputX, useDropout, inputDropoutRate, false, 0.0);
        layers[0][0] = inputNode;


        //first hidden layer has 4 nodes
        layers[1] = new ConvolutionalNode[4];
        for (int i = 0; i < 4; i++) {
            ConvolutionalNode node = new ConvolutionalNode(1, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 20, 20, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[1][i] = node;

            numberWeights += node.getNumberWeights();

            new ConvolutionalEdge(layers[0][0], node, inputChannels /* will be 1 for MNIST or 3 for CIFAR-10 to get the other feature maps to only have 1 channel*/, 13, 13);

            numberWeights += inputChannels * 13 * 13;
        }


        //second hidden layer also has 4 nodes, because it's just the max pooling from the first
        layers[2] = new ConvolutionalNode[4];
        for (int i = 0; i < 4; i++) {
            ConvolutionalNode node = new ConvolutionalNode(2, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 10, 10, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[2][i] = node;

            numberWeights += node.getNumberWeights();

            new PoolingEdge(layers[1][i], node, 2, 2); //stride of 2 and pool size of 2 for this max pooling operation
        }


        //third hidden layer is dense with 10 nodes
        layers[3] = new ConvolutionalNode[10];
        for (int i = 0; i < 10; i++) {
            ConvolutionalNode node = new ConvolutionalNode(3, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[3][i] = node;

            numberWeights += node.getNumberWeights();

            for (int j = 0; j < 4; j++) {
                new ConvolutionalEdge(layers[2][j], node, 1, 10, 10);

                numberWeights += 10 * 10;
            }
        }

        //output layer is dense with 10 nodes
        layers[4] = new ConvolutionalNode[10];
        for (int i = 0; i < 10; i++) {
            //no dropout or BN on output nodes
            ConvolutionalNode node = new ConvolutionalNode(4, i, NodeType.OUTPUT, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, false, 0.0, false, 0.0);
            layers[4][i] = node;

            //no bias on output nodes

            for (int j = 0; j < 10; j++) {
                new ConvolutionalEdge(layers[3][j], node, 1, 1, 1);

                numberWeights += 1;
            }
        }
    }

    public void createLeNet5(ActivationType activationType, int batchSize, int inputChannels, int inputY, int inputX, int inputPadding, int outputLayerSize) throws NeuralNetworkException {
        //TODO: Programming Assignment 3 - Part 1: Implement creating a LeNet-5 CNN
        //make sure dropout is turned off on the last hidden layer

        //There are 8 layers in LeNet5
        layers = new ConvolutionalNode[8][];

        //input layer has 1 node
        layers[0] = new ConvolutionalNode[1];
        ConvolutionalNode inputNode = new ConvolutionalNode(0, 0, NodeType.INPUT, activationType /*doesn't matter for input*/, inputPadding /*padding*/ , batchSize, inputChannels, inputY, inputX, useDropout, inputDropoutRate, false, 0.0);
        layers[0][0] = inputNode;


        //first hidden layer has 6 nodes
        layers[1] = new ConvolutionalNode[6];
        for (int i = 0; i < layers[1].length ; i++) {
            ConvolutionalNode node = new ConvolutionalNode(1, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 28, 28, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[1][i] = node;

            numberWeights += node.getNumberWeights();

            new ConvolutionalEdge(layers[0][0], node, inputChannels /* will be 1 for MNIST or 3 for CIFAR-10 to get the other feature maps to only have 1 channel*/, 5, 5);

            numberWeights += inputChannels * 5 * 5;
        }


        //second hidden layer also has 6 nodes, because it's just the max pooling from the first
        layers[2] = new ConvolutionalNode[6];
        for (int i = 0; i < layers[2].length; i++) {
            ConvolutionalNode node = new ConvolutionalNode(2, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 14, 14, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[2][i] = node;

            numberWeights += node.getNumberWeights();

            new PoolingEdge(layers[1][i], node, 2, 2);
        }

        //third hidden layer has 16 nodes and partially connected
        layers[3] = new ConvolutionalNode[16];

        //partial_edges_three = 6; 6x3 = 18 edges
        //partial_edges_four = 9; 9x4 = 36 edges(6 rules, 3 manual)
        //partial_edges_six = 1; 6x1 = 6 edges

        for (int i = 0; i < layers[3].length ; i++) {
            ConvolutionalNode node = new ConvolutionalNode(3, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 10, 10, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[3][i] = node;

            numberWeights += node.getNumberWeights();

            if (i < 6) {
                //connect # of nodes 3
                for (int j=0; j <3; j++){
                    if (i+j < layers[2].length){
                        new ConvolutionalEdge(layers[2][i + j], node,1, 5, 5);
                    }else{
                        new ConvolutionalEdge(layers[2][i + j - 6], node, 1, 5, 5);
                        }
                    numberWeights += 5 * 5;
                    }
            }else if (i < 12){
                //connect # of nodes 4
                for (int j=0; j <4; j++){
                    if ( (i-6)+j < layers[2].length){
                        new ConvolutionalEdge(layers[2][(i-6) + j], node,1, 5, 5);
                    }else{
                        new ConvolutionalEdge(layers[2][(i-6) + j -6], node, 1, 5, 5);
                        }
                        numberWeights += 5 * 5;
                    }
            }else if(i < 15){
                switch (i){

                    //0,1,3,4
                    case 12:
                            new ConvolutionalEdge(layers[2][0],node,1,5,5);
                            new ConvolutionalEdge(layers[2][1],node,1,5,5);
                            new ConvolutionalEdge(layers[2][3],node,1,5,5);
                            new ConvolutionalEdge(layers[2][4],node,1,5,5);
                            numberWeights += 5 * 5 * 4;
                            break;
                    //1,2,4,5
                    case 13:
                            new ConvolutionalEdge(layers[2][1],node,1,5,5);
                            new ConvolutionalEdge(layers[2][2],node,1,5,5);
                            new ConvolutionalEdge(layers[2][4],node,1,5,5);
                            new ConvolutionalEdge(layers[2][5],node,1,5,5);
                            numberWeights += 5 * 5 * 4;
                            break;
                    //0,2,3,5
                    case 14:
                            new ConvolutionalEdge(layers[2][0],node,1,5,5);
                            new ConvolutionalEdge(layers[2][2],node,1,5,5);
                            new ConvolutionalEdge(layers[2][3],node,1,5,5);
                            new ConvolutionalEdge(layers[2][5],node,1,5,5);
                            numberWeights += 5 * 5 * 4;
                            break;
                    default:
                            break;
                    }
            }else{
                //connect # of nodes 6
                for (int j=0; j <6; j++){
                    new ConvolutionalEdge(layers[2][j], node,1, 5, 5);
                    numberWeights += 5 * 5;
                }
            }
        }


        //four hidden layer also has 16 nodes, because it's just the max pooling from the first
        layers[4] = new ConvolutionalNode[16];
        for (int i = 0; i < layers[4].length; i++) {
            ConvolutionalNode node = new ConvolutionalNode(4, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 5, 5, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[4][i] = node;

            numberWeights += node.getNumberWeights();

            new PoolingEdge(layers[3][i], node, 2, 2);
        }




        //five hidden layer is dense with 120 nodes
        layers[5] = new ConvolutionalNode[120];
        for (int i = 0; i < layers[5].length; i++) {
            ConvolutionalNode node = new ConvolutionalNode(5, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[5][i] = node;

            numberWeights += node.getNumberWeights();

            for (int j = 0; j < 16; j++) {
                new ConvolutionalEdge(layers[4][j], node, 1, 5, 5);

                numberWeights += 5 * 5;
            }
        }

        //six hidden layer is dense with 84 nodes
        layers[6] = new ConvolutionalNode[84];
        for (int i = 0; i < layers[6].length; i++) {
            //useDropout turn it off when doing PA33TestCnn
            ConvolutionalNode node = new ConvolutionalNode(6, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, false, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[6][i] = node;

            numberWeights += node.getNumberWeights();

            for (int j = 0; j < 120; j++) {
                new ConvolutionalEdge(layers[5][j], node, 1, 1, 1);

                numberWeights += 1 * 1;
            }
        }
        //output layer is dense with 10 nodes
        layers[7] = new ConvolutionalNode[10];
        for (int i = 0; i < layers[7].length; i++) {
            ConvolutionalNode node = new ConvolutionalNode(7, i, NodeType.OUTPUT, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, false, 0.0, false, 0.0);
            layers[7][i] = node;

            //no bias in output layer

            for (int j = 0; j < 84; j++) {
                new ConvolutionalEdge(layers[6][j], node, 1, 1, 1);

                numberWeights += 1 * 1;
            }
        }
    }

    public ConvolutionalNeuralNetwork(LossFunction lossFunction, boolean useDropout, double inputDropoutRate, double hiddenDropoutRate, boolean useBatchNormalization, double alpha) {
        this.lossFunction = lossFunction;
        this.useDropout = useDropout;
        this.inputDropoutRate = inputDropoutRate;
        this.hiddenDropoutRate = hiddenDropoutRate;
        this.useBatchNormalization = useBatchNormalization;
        this.alpha = alpha;
    }

    /**
     * This gets the number of weights in the ConvolutionalNeuralNetwork, which should
     * be equal to the number of hidden nodes (1 bias per hidden node) plus
     * the number of edges (1 bias per edge). It is updated whenever an edge
     * is added to the neural network.
     *
     * @return the number of weights in the neural network.
     */
    public int getNumberWeights() {
        return numberWeights;
    }

    /**
     * This resets all the values that are modified in the forward pass and
     * backward pass and need to be reset to 0 before doing another
     * forward and backward pass (i.e., all the non-weights/biases).
     */
    public void reset() {
        for (int layer = 0; layer < layers.length; layer++) {
            for (int number = 0; number < layers[layer].length; number++) {
                //call reset on each node in the network
                layers[layer][number].reset();
            }
        }
    }

    /**
     * This resets the running averages for batch normalization
     * across all the nodes at the beginning of an epoch.
     */
    public void resetRunning() {
        for (int layer = 0; layer < layers.length; layer++) {
            for (int number = 0; number < layers[layer].length; number++) {
                //call reset on each node in the network
                layers[layer][number].resetRunning();
            }
        }
    }

    /**
     * This returns an array of every weight (including biases) in the ConvolutionalNeuralNetwork.
     * This will be very useful in backpropagation and sanity checking.
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public double[] getWeights() throws NeuralNetworkException {
        double[] weights = new double[numberWeights];

        //What we're going to do here is fill in the weights array
        //we just created by having each node set the weights starting
        //at the position variable we're creating. The Node.getWeights
        //method will set the weights variable passed as a parameter,
        //and then return the number of weights it set. We can then
        //use this to increment position so the next node gets weights
        //and puts them in the right position in the weights array.
        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nWeights = layers[layer][nodeNumber].getWeights(position, weights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the ConvolutionalNeuralNetwork was (" + numberWeights + ") but when getting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return weights;
    }

    /**
     * This sets every weight (including biases) in the ConvolutionalNeuralNetwork, it sets them in
     * the same order that they are retreived by the getWeights method.
     * This will be very useful in backpropagation and sanity checking.
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public void setWeights(double[] newWeights) throws NeuralNetworkException {
        if (numberWeights != newWeights.length) {
            throw new NeuralNetworkException("Could not setWeights because the number of new weights: " + newWeights.length + " was not equal to the number of weights in the ConvolutionalNeuralNetwork: " + numberWeights);
        }

        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                Log.trace("setting weights for layer: " + layer + ", nodeNumber: " + nodeNumber + ", position: " + position);
                int nWeights = layers[layer][nodeNumber].setWeights(position, newWeights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the ConvolutionalNeuralNetwork was (" + numberWeights + ") but when setting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }
    }

    /**
     * This returns an array of every weight (including biases) in the ConvolutionalNeuralNetwork.
     * This will be very useful in backpropagation and sanity checking.
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public double[] getDeltas() throws NeuralNetworkException {
        double[] deltas = new double[numberWeights];

        //What we're going to do here is fill in the deltas array
        //we just created by having each node set the deltas starting
        //at the position variable we're creating. The Node.getDeltas
        //method will set the deltas variable passed as a parameter,
        //and then return the number of deltas it set. We can then
        //use this to increment position so the next node gets deltas
        //and puts them in the right position in the deltas array.
        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nDeltas = layers[layer][nodeNumber].getDeltas(position, deltas);
                position += nDeltas;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the ConvolutionalNeuralNetwork was (" + numberWeights + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return deltas;
    }


    /**
     * This initializes the weights in the CNN using either Xavier or
     * Kaiming initialization.
    *
     * @param type will be either "xavier" or "kaiming" and this will
     * initialize the child nodes accordingly, using their helper methods.
     * @param bias is the value to set the bias of each node to.
     */
    public void initializeRandomly(String type, double bias) {
        //TODO: You need to implement this for Programming Assignment 3 - Part 1
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                ConvolutionalNode node = layers[layer][nodeNumber];

                if ( type.equals("xavier") ){
                    //get the fanin weights and fanout weights
                    int fanIn = 0;
                    int fanOut = 0;

                    for (Edge edge_in: node.inputEdges){
                        double weights[] = new double[edge_in.sizeZ * edge_in.sizeY * edge_in.sizeX];
                        fanIn += edge_in.getWeights(0 , weights);
                        }

                    for (Edge edge_out: node.outputEdges){
                        double weights[] = new double[edge_out.sizeZ * edge_out.sizeY * edge_out.sizeX];
                        fanOut += edge_out.getWeights(0, weights);
                        }

                    node.initializeWeightsAndBiasXavier(bias, fanIn, fanOut);
                }else{
                    //get the fanin weights and fanout weights
                    int fanIn = 0;

                    for (Edge edge_in: node.inputEdges){
                        double weights[] = new double[edge_in.sizeZ * edge_in.sizeY * edge_in.sizeX];
                        fanIn += edge_in.getWeights(0 , weights);
                        }

                    node.initializeWeightsAndBiasKaiming(bias, fanIn);

                    }

                }
            }
    }



    /**
     * This performs a forward pass through the neural network given
     * inputs from the input instance.
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param startIndex is the index of the first image to use
     * @param batchSize is the number of images to use
     *
     * @return the sum of the output of all output nodes
     */
    public double forwardPass(ImageDataSet imageDataSet, int startIndex, int batchSize, boolean training) throws NeuralNetworkException {
        //be sure to reset before doing a forward pass
        reset();

        List<Image> images = imageDataSet.getImages(startIndex, batchSize);

        for (int number = 0; number < layers[0].length; number++) {
            ConvolutionalNode inputNode = layers[0][number];
            inputNode.setValues(images, imageDataSet.getChannelAvgs(), imageDataSet.getChannelStdDevs(imageDataSet.getChannelAvgs()));
        }

        //TODO: You need to implement propagating forward for each node
        //for Programming Assignment 3 - Part 1

        //The following is needed for Programming Assignment 3 - Part 1
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                ConvolutionalNode node = layers[layer][nodeNumber];
                node.propagateForward(training);
                }
            }

        double lossSum = 0;
        double output[][] = this.getOutputValues(batchSize);

        if (lossFunction == LossFunction.SVM) {
            //TODO: Implement this for Programming Assignment 3 - Part 1, be sure
            //to calculate for each image in the batch

            for (int i=0; i < images.size(); i++){
                double error[] = new double[nOutputs];
                int position = images.get(i).label;
                int yDelta = 0;

                //calculate the error
                for(int k=0; k < nOutputs; k++){

                    if(k == position){
                        error[k] = 0.0;
                    }else{
                        error[k] = Math.max(0, output[i][k] - output[i][position] + 1) ;
                        }
                    }

                //sum all the error
                for (double number: error){
                    lossSum += number;
                    if (number >0){
                        yDelta++;
                        }
                    }

                //set all the delta
                for(int j=0; j < nOutputs; j++){
                    ConvolutionalNode outputNode = layers[outputLayer][j];

                    if(j == position){
                        outputNode.delta[i][0][0][0] = -yDelta;
                    }else{
                        if (error[j] >0){
                            outputNode.delta[i][0][0][0] = 1;
                        }else{
                            outputNode.delta[i][0][0][0] = 0;
                        }
                        }
                    }
                }
        } else if (lossFunction == LossFunction.SOFTMAX) {
            //TODO: Implement this for Programming Assignment 3 - Part 1, be sure
            //to calculate for each image in the batch
            for (int i =0; i < images.size(); i++){
                double sumPost = 0.0;
                int position = images.get(i).label;
                double expected = 0.0;

                //calculate the error and sum all the ezi
                for (int n =0; n < nOutputs; n++){
                        ConvolutionalNode outputNode = layers[outputLayer][n];
                        if (position == n){
                            expected = Math.exp(outputNode.outputValues[i][0][0][0]);
                            }
                        double loss = Math.exp(outputNode.outputValues[i][0][0][0]);
                        sumPost += loss;
                        outputNode.delta[i][0][0][0] = loss;
                    }

                lossSum += -Math.log(expected / sumPost);
                //update the delta and sum the loss value
                for (int t=0; t < nOutputs; t++){

                    ConvolutionalNode outputNode = layers[outputLayer][t];

                    if (t == position){
                        outputNode.delta[i][0][0][0] =  outputNode.delta[i][0][0][0]/ sumPost -1;
                    }else{
                        outputNode.delta[i][0][0][0] = outputNode.delta[i][0][0][0] / sumPost;
                        }
                    }
                }
        } else {
            throw new NeuralNetworkException("Could not do a CharacterSequence forward pass on ConvolutionalNeuralNetwork because lossFunction was unknown or invalid: " + lossFunction);
        }

        return lossSum;
    }

    /**
     * This does forward passes over the entire image data set to calculate
     * the total error and accuracy (this is used by GradientDescent.java). We
     * do them both here to improve performance.
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param accuracyAndError is a double array of length 2, index 0 will
     * be the accuracy and index 1 will be the error
     */
    public void calculateAccuracyAndError(ImageDataSet imageDataSet, int batchSize, double[] accuracyAndError) throws NeuralNetworkException {
        //TODO: need to implement this for Programming Assignment 3 - Part 2
        //the output node with the maximum value is the predicted class
        //you need to sum up how many of these match the actual class
        //for each imageDataSet, and then calculate:
        //num correct / total to get a percentage accuracy

        accuracyAndError[0] = 0.0;
        accuracyAndError[1] = 0.0;

        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        int total_images = imageDataSet.getNumberImages();
        int correct = 0;
        boolean training = false;//true, do training version of batch norm. false, do inference version of batch norm.



        for (int startIndex = 0; startIndex < total_images ; startIndex += batchSize){
            accuracyAndError[1] += this.forwardPass(imageDataSet, startIndex, batchSize, training);
            for (int i = 0; i < batchSize; i++){
                //get the correct class
                int position = imageDataSet.getImages(startIndex + i, 1).get(0).label;
                int predictedClass = 0;

                //do the forward pass and get the outputs
                //double output[][] = this.getOutputValues(batchSize);

                //find the largest output value
                double maxValue = -Double.MAX_VALUE;
                for (int j=0; j < nOutputs; j++){

                    double outputVal = layers[outputLayer][j].outputValues[i][0][0][0];
                    if (outputVal > maxValue){
                        predictedClass = j;
                        maxValue = outputVal;
                        }
                    }

                if (predictedClass == position){
                    correct++;
                    }
                }
            }

        accuracyAndError[0] =  (1.0 * correct) * (1.0 / total_images);
    }


    /**
     * This gets the output values of the neural network
     * after a forward pass, this will be a 1 dimensional array, one
     * value for each output node
     *
     * @param batchSize is the batch size of for this CNN
     *
     * @return a one dimensional array of the output values from this neural network for
     */
    public double[][] getOutputValues(int batchSize) {
        //the number of output values is the number of output nodes
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        double[][] outputValues = new double[batchSize][nOutputs];

        for (int i = 0; i < batchSize; i++) {
            for (int number = 0; number < nOutputs; number++) {
                outputValues[i][number] = layers[outputLayer][number].outputValues[i][0][0][0];
            }
        }

        return outputValues;
    }

    /**
     * The step size used to calculate the gradient numerically using the finite
     * difference method.
     */
    private static final double H = 0.0000001;

    /**
     * This calculates the gradient of the neural network with it's current
     * weights for a given DataSet Instance using the finite difference method:
     * gradient[i] = (f(x where x[i] = x[i] + H) - f(x where x[i] = x[i] - H)) / 2h
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param startIndex is the index of the first image to use
     * @param batchSize is the number of images to use
     */
    public double[] getNumericGradient(ImageDataSet imageDataSet, int startIndex, int batchSize) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 3 - Part 2

            boolean training = true;//true, do training version of batch norm. false, do inference version of batch norm.
            double originalWeights[] = this.getWeights();
            double[] gradient = new double[this.numberWeights];
            double tempWeights[] = this.getWeights();

            for (int i=0; i< this.numberWeights; i++){
                double output1 = 0.0;
                double output2 = 0.0;


                tempWeights[i] = originalWeights[i] + H;
                this.setWeights(tempWeights);
                output1 = this.forwardPass(imageDataSet, startIndex, batchSize, true);

                tempWeights[i] = originalWeights[i] - H;
                this.setWeights(tempWeights);
                output2 = this.forwardPass(imageDataSet, startIndex, batchSize, true);

                gradient[i] = (output1 - output2)/ (2*H);
                tempWeights[i] = originalWeights[i];
                }

            this.setWeights(originalWeights);
            return gradient;
        //throw new NeuralNetworkException("ConvolutionalNeuralNetwork.getNumericGradient not implemented!");
    }


    /**
     * This performs a backward pass through the neural network given
     * outputs from the given instance. This will set the deltas in
     * all the edges and nodes which will be used to calculate the
     * gradient and perform backpropagation.
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param startIndex is the index of the first image to use
     * @param batchSize is the number of images to use
     *
     */
    public void backwardPass(ImageDataSet imageDataSet, int startIndex, int batchSize) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 3 - Part 2
        int OutputLayer = layers.length -1;

        for (int number = OutputLayer; number >= 0 ; number--) {
            for (int pos =0; pos < layers[number].length; pos++){
                ConvolutionalNode outputNode =  layers[number][pos];
                outputNode.propagateBackward();
                }
            }
    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g.,
     * the ConvolutionalNeuralNetwork.backwardPass(imageDataSetm startIndex, batchSize)) Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient).
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param startIndex is the index of the first image to use
     * @param batchSize is the number of images to use
     */
    public double[] getGradient(ImageDataSet imageDataSet, int startIndex, int batchSize) throws NeuralNetworkException {
        forwardPass(imageDataSet, startIndex, batchSize, true /*we're training here so use the training versions of batch norm and dropout*/);
        backwardPass(imageDataSet, startIndex, batchSize);

        return getDeltas();
    }

    /**
     * Print out numeric vs backprop gradients in a clean manner so that
     * you can see where gradients were not the same
     *
     * @param numericGradient is a previously calculated numeric gradient
     * @param backpropGradient is a previously calculated gradient from backprop
     */
    public void printGradients(double[] numericGradient, double[] backpropGradient) {
        int current = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int number = 0; number < layers[layer].length; number++) {
                //call reset on each node in the network
                current += layers[layer][number].printGradients(current, numericGradient, backpropGradient);
            }
        }
    }
}
