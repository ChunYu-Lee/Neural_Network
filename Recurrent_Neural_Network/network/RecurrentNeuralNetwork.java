package network;

import java.util.List;

import data.Sequence;
import data.CharacterSequence;
import data.TimeSeries;

import util.Log;


public class RecurrentNeuralNetwork {
    //this is the loss function for the output of the neural network
    LossFunction lossFunction;
    
    //this is the maximum length of any sequence
    int maxSequenceLength;

    //this is the total number of weights in the neural network
    int numberWeights;
    
    //layers contains all the nodes in the neural network
    RecurrentNode[][] layers;

    public RecurrentNeuralNetwork(int inputLayerSize, int[] hiddenLayerSizes, int outputLayerSize, int maxSequenceLength, RNNNodeType rnnNodeType, LossFunction lossFunction) {
        this.lossFunction = lossFunction;
        this.maxSequenceLength = maxSequenceLength;

        //the number of layers in the neural network is 2 plus the number of hidden layers,
        //one additional for the input, and one additional for the output.

        //create the outer array of the 2-dimensional array of nodes
        layers = new RecurrentNode[hiddenLayerSizes.length + 2][];

        //we will progressively calculate the number of weights as we create the network. the
        //number of edges will be equal to the number of hidden nodes (each has a bias weight, but
        //the input and output nodes do not) plus the number of edges
        numberWeights = 0;

        Log.info("creating a neural network with " + hiddenLayerSizes.length + " hidden layers, for max sequence length: " + maxSequenceLength + ".");
        for (int layer = 0; layer < layers.length; layer++) {
            
            //determine the layer size depending on the layer number, 0 is the
            //input layer, and the last layer is the output layer, all others
            //are hidden layers
            int layerSize;
            NodeType nodeType;
            if (layer == 0) {
                //this is the input layer
                layerSize = inputLayerSize;
                nodeType = NodeType.INPUT;
                Log.info("input layer " + layer + " has " + layerSize + " nodes.");

            } else if (layer < layers.length - 1) {
                //this is a hidden layer
                layerSize = hiddenLayerSizes[layer - 1];
                nodeType = NodeType.HIDDEN;
                Log.info("hidden layer " + layer + " has " + layerSize + " nodes.");

            } else {
                //this is the output layer
                layerSize = outputLayerSize;
                nodeType = NodeType.OUTPUT;
                Log.info("output layer " + layer + " has " + layerSize + " nodes.");
            }

            //create the layer with the right length and right node types
            layers[layer] = new RecurrentNode[layerSize];

            for (int j = 0; j < layers[layer].length; j++) {
                if (nodeType == NodeType.INPUT) {
                    //input nodes do not have an activation function applied to them
                    layers[layer][j] = new RecurrentNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength, ActivationType.LINEAR);

                } else {
                    switch (rnnNodeType) {
                        case LINEAR:
                            layers[layer][j] = new RecurrentNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength, ActivationType.LINEAR);
                            //increment the number of weights here because some node types will have a different amount
                            //linear nodes don't have bias in the output layer
                            if (nodeType == NodeType.HIDDEN) numberWeights++;
                            break;
                        case SIGMOID:
                            layers[layer][j] = new RecurrentNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength, ActivationType.SIGMOID);
                            //increment the number of weights here because some node types will have a different amount
                            //sigmoid nodes don't have bias in the output layer
                            if (nodeType == NodeType.HIDDEN) numberWeights++;
                            break;
                        case TANH:
                            layers[layer][j] = new RecurrentNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength, ActivationType.TANH);
                            //increment the number of weights here because some node types will have a different amount
                            //tanh nodes don't have bias in the output layer
                            if (nodeType == NodeType.HIDDEN) numberWeights++;
                            break;
                        case LSTM:
                            //TODO: implement this for Programming Assignment 2 - Part 4
                            //increment the number of weights here because some node types will have a different amount
                            layers[layer][j] = new LSTMNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength);
                            numberWeights += 11;
                            break;
                        case GRU:
                            //TODO: BONUS implement this for Programming Assignment 2 - Part 4
                            layers[layer][j] = new GRUNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength);
                            numberWeights += 9;
                            break;
                        case MGU:
                            //TODO: BONUS implement this for Programming Assignment 2 - Part 4
                            layers[layer][j] = new MGUNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength);
                            numberWeights += 6;
                            break;
                        case UGRNN:
                            //TODO: BONUS implement this for Programming Assignment 2 - Part 4
                            layers[layer][j] = new UGRNNNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength);
                            numberWeights += 6;
                            break;
                        case DELTA:
                            //TODO: BONUS implement this for Programming Assignment 2 - Part 4
                            layers[layer][j] = new DeltaNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength);
                            numberWeights += 6;
                            break;
                        default:
                            Log.fatal("Trying to create RNN with unknown RNNNodeType - this should never happen!");
                            System.exit(1);
                    }
                }
            }
        }
    }

    /**
     * This gets the number of weights in the RecurrentNeuralNetwork, which should
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
     * Gets the name for each weight so things can be debugged easier
     * @return a array of strings, each corresponding to what edge/bias each weight represents from the
     *      results of the getWeights() method.
     */
    public String[] getWeightNames() throws NeuralNetworkException {
        String[] weightNames = new String[numberWeights];

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
                int nWeights = layers[layer][nodeNumber].getWeightNames(position, weightNames);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the RecurrentNeuralNetwork was (" + numberWeights + ") but when getting the weight names there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return weightNames;
    }


    /**
     * This returns an array of every weight (including biases) in the RecurrentNeuralNetwork.
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
                    throw new NeuralNetworkException("The numberWeights field of the RecurrentNeuralNetwork was (" + numberWeights + ") but when getting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return weights;
    }

    /**
     * This sets every weight (including biases) in the RecurrentNeuralNetwork, it sets them in
     * the same order that they are retreived by the getWeights method.
     * This will be very useful in backpropagation and sanity checking. 
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public void setWeights(double[] newWeights) throws NeuralNetworkException {
        if (numberWeights != newWeights.length) {
            throw new NeuralNetworkException("Could not setWeights because the number of new weights: " + newWeights.length + " was not equal to the number of weights in the RecurrentNeuralNetwork: " + numberWeights);
        }

        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nWeights = layers[layer][nodeNumber].setWeights(position, newWeights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the RecurrentNeuralNetwork was (" + numberWeights + ") but when setting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }
    }

    /**
     * This returns an array of every weight (including biases) in the RecurrentNeuralNetwork.
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
                    throw new NeuralNetworkException("The numberWeights field of the RecurrentNeuralNetwork was (" + numberWeights + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return deltas;
    }


    /**
     * This adds edges to the RecurrentNeuralNetwork, connecting each node
     * in a layer to each node in the subsequent layer
     */
    public void connectFully() throws NeuralNetworkException {
        //create outgoing edges from the input layer to the last hidden layer,
        //the output layer will not have outgoing edges
        for (int layer = 0; layer < layers.length - 1; layer++) {

            //iterate over the nodes in the current layer
            for (int inputNodeNumber = 0; inputNodeNumber < layers[layer].length; inputNodeNumber++) {

                //iterate over the nodes in the next layer
                for (int outputNodeNumber = 0; outputNodeNumber < layers[layer + 1].length; outputNodeNumber++) {
                    RecurrentNode inputNode = layers[layer][inputNodeNumber];
                    RecurrentNode outputNode = layers[layer + 1][outputNodeNumber];
                    new Edge(inputNode, outputNode);

                    //as we added an edge, the number of weights should increase by 1
                    numberWeights++;
                    Log.trace("numberWeights now: " + numberWeights);
                }
            }
        }
    }

    /**
     * Makes this RNN a Jordan network by creating a RecurrentEdge between
     * every output node and every hidden node.
     *
     * @param timeSkip is how many time steps to skip for the recurrent connection
     */
    public void connectJordan(int timeSkip) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 2 - Part 1
        int outputLayer = layers.length -1; 
        //for (int i=1; i < timeSkip +1; i++){
            
            //loop the hidden layers
            for (int j=1; j < layers.length -1; j++){
                
                for(int k=0; k < layers[j].length; k++){
                    RecurrentNode outputnode = layers[j][k];

                    for (int l=0; l < layers[outputLayer].length; l++){
                        new RecurrentEdge(layers[outputLayer][l], outputnode, timeSkip);
                        this.numberWeights++;
                        Log.trace("numberWeights now: " + numberWeights);
                        }
                    }
                }
            //}
    }

    /**
     * Makes this RNN an Elman network by creating a RecurrentEdge from 
     * every hidden node in a layer to every other hidden node in that layer
     *
     * @param timeSkip is how many time steps to skip for the recurrent connection
     */
    public void connectElman(int timeSkip) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 2 - Part 1
         
        for (int i=1; i < timeSkip +1; i++){
            
            //loop the hidden layers
            for (int j=1; j <= layers.length -2; j++){
                
                //link every hidden node in this layer to all the other nodes in that layer
                for (RecurrentNode inputnode: layers[j]){
                    for (RecurrentNode outputnode: layers[j]){
                        new RecurrentEdge(inputnode, outputnode, timeSkip);
                        this.numberWeights ++;
                        }
                    }
                }
            }
    
    }


    /**
     * This will create an Edge between the node with number inputNumber on the inputLayer to the
     * node with the outputNumber on the outputLayer.
     *
     * @param inputLayer the layer of the input node
     * @param inputNumber the number of the input node on layer inputLayer
     * @param outputLayer the layer of the output node
     * @param outputNumber the number of the output node on layer outputLayer
     */
    public void connectNodes(int inputLayer, int inputNumber, int outputLayer, int outputNumber) throws NeuralNetworkException {
        if (inputLayer >= outputLayer) {
            throw new NeuralNetworkException("Cannot create an Edge between input layer " + inputLayer + " and output layer " + outputLayer + " because the layer of the input node must be less than the layer of the output node.");
        //} else if (outputLayer != inputLayer + 1) {
            //throw new NeuralNetworkException("Cannot create an Edge between input layer " + inputLayer + " and output layer " + outputLayer + " because the layer of the output node must be the next layer in the network.");
        }

        //TODO: Complete this function. BONUS: allow it to it create edges that can skip layers
        //input node
        RecurrentNode inputnode = layers[inputLayer][inputNumber];
        //output node
        RecurrentNode outputnode = layers[outputLayer][outputNumber];

        //set up the edge
        Edge edge = new Edge(inputnode, outputnode);
        
        inputnode.addOutgoingEdge(edge);
        outputnode.addIncomingEdge(edge);
        
        this.numberWeights++;
    }

    /**
     * This initializes the weights in the RNN using either Xavier or
     * Kaiming initialization.
    *
     * @param type will be either "xavier" or "kaiming" and this will
     * initialize the child nodes accordingly, using their helper methods.
     * @param bias is the value to set the bias of each node to.
     */
    public void initializeRandomly(String type, double bias) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        for (int i=0; i < layers.length; i++){ 
            for (RecurrentNode recurrentNode: layers[i]){
                //fanIn = inputEdges + RecurrentEdges
                int fanIn = recurrentNode.inputEdges.size() + recurrentNode.inputRecurrentEdges.size();
                if (type.equals("xavier")){
                    int fanOut = recurrentNode.outputEdges.size() + recurrentNode.outputRecurrentEdges.size();
                    recurrentNode.initializeWeightsAndBiasXavier(fanIn, fanOut, bias);
                }else if(type.equals("kaiming")){
                    recurrentNode.initializeWeightsAndBiasKaiming(fanIn, bias);
                }  
            }
        }
    }



    /**
     * This performs a forward pass through the neural network given
     * inputs from the input instance.
     *
     * @param instance is the data set instance to pass through the network
     *
     * @return the sum of the output of all output nodes
     */
    public double forwardPass(Sequence sequence) throws NeuralNetworkException {
        //be sure to reset before doing a forward pass
        reset();

        //set input values differently for time series and character sequences
        if (sequence instanceof CharacterSequence) {
            CharacterSequence characterSequence = (CharacterSequence)sequence;

            //set the input nodes for each time step in the CharacterSequence
            for (int timeStep = 0; timeStep < characterSequence.getLength() - 1; timeStep++) {
                int value = characterSequence.valueAt(timeStep);
                for (int number = 0; number < layers[0].length; number++) {
                    RecurrentNode inputNode = layers[0][number];
                    if (value == number) {
                        //the value will be 0..n where n is the number of input nodes (and possible character values)
                        inputNode.postActivationValue[timeStep] = 1.0;
                    } else {
                        inputNode.postActivationValue[timeStep] = 0.0;
                    }
                }
            }

        } else if (sequence instanceof TimeSeries) {
            TimeSeries series = (TimeSeries)sequence;

            //set the input nodes for each time step in the TimeSeries
            for (int timeStep = 0; timeStep < series.getLength() - 1; timeStep++) {
                for (int number = 0; number < layers[0].length; number++) {
                    RecurrentNode inputNode = layers[0][number];
                    inputNode.postActivationValue[timeStep] = series.getInputValue(timeStep, number);
                }
            }

        }

        //TODO: You need to implement propagating forward for each node (output nodes need
        //to be propagated forward for their recurrent connections to further time steps)
        //for Programming Assignment 2 - Part 1
        //NOTE: This shouldn't need to be changed for Programming Assignment 2 - Parts 2 or 3

        for (int timeStep = 0; timeStep < sequence.getLength()-1; timeStep++){
            for (int i=0; i < layers.length; i++){
                for (RecurrentNode node: layers[i]){
                    node.propagateForward(timeStep);
                    }
                }
            }



        //The following is needed for Programming Assignment 2 - Part 1
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        //note that the target value for any time step is the sequence value at that time step + 1
        //this means you should only go up to length - 1 time steps in calculating the loss
        double lossSum = 0;
        if (sequence instanceof CharacterSequence) {
            //calculate the loss functions for character sequences
            CharacterSequence characterSequence = (CharacterSequence)sequence;
            
                if (lossFunction == LossFunction.NONE) {
                    //for no loss function we can just return the sum of the outputs overall time steps, and can set each output node's delta to 1
                    for (int timeStep = 0; timeStep < characterSequence.getLength() - 1; timeStep++) {
                        for (int number = 0; number < nOutputs; number++) {
                            RecurrentNode outputNode = layers[outputLayer][number];
                            lossSum += outputNode.postActivationValue[timeStep];
                            outputNode.delta[timeStep] = 1.0;
                        }
                    }

                }else if(lossFunction == LossFunction.SVM){ 
                    //TODO: Implement this for Programming Assignment 2 - Part 1
                    for (int timeStep = 0; timeStep < characterSequence.getLength() -1; timeStep++){
                        
                        int expectedEncoding = (int) characterSequence.valueAt(timeStep + 1);
                        int yDelta = 0;
                        double error[] = new double[nOutputs];

                        //calculate the error
                        for(int i=0; i<nOutputs; i++){
                            RecurrentNode outputNode = layers[outputLayer][i];
                            if(i == expectedEncoding){
                                error[i] = 0.0;
                            }else{
                                error[i] = Math.max(0, outputNode.postActivationValue[timeStep] - layers[outputLayer][expectedEncoding].postActivationValue[timeStep] + 1) ; 
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
                            RecurrentNode outputNode = layers[outputLayer][j];
                
                            if(j == expectedEncoding){
                                outputNode.delta[timeStep] = -yDelta;
                            }else{
                                if (error[j] >0){
                                    outputNode.delta[timeStep] = 1;
                                }else{
                                    outputNode.delta[timeStep] = 0;
                                    }
                                }
                            }
                        }
                } else if (lossFunction == LossFunction.SOFTMAX) {
                    //TODO: Implement this for Programming Assignment 2 - Part 1
                    
                    for (int timeStep = 0; timeStep < characterSequence.getLength() -1; timeStep++){
                        
                        double error[] = new double[nOutputs];
                        double sumPost = 0.0;
                        int expectedTarget = (int) characterSequence.valueAt(timeStep +1);
                        double maxValue = layers[outputLayer][0].postActivationValue[timeStep];

                        //get the max value and position
                        for (int number = 0; number < nOutputs; number++) {
                    
                            RecurrentNode outputNode = layers[outputLayer][number]; 
                            if (outputNode.postActivationValue[timeStep] > maxValue){
                                maxValue = outputNode.postActivationValue[timeStep];
                                } 

                            }
                        //calculate the error and sum all the ezi
                        for (int n =0; n < nOutputs; n++){
                    
                            RecurrentNode outputNode = layers[outputLayer][n];

                            error[n] = Math.exp(outputNode.postActivationValue[timeStep] - maxValue);
                            sumPost += error[n];
                            }
                
                        //update the delta and sum the loss value 
                        for (int t=0; t < nOutputs; t++){
                    
                            RecurrentNode outputNode = layers[outputLayer][t];
                    
                            if (t == expectedTarget){
                                outputNode.delta[timeStep] =  (error[t] / sumPost) -1;
                                lossSum += -Math.log(error[t] / sumPost);
                            }else{
                                outputNode.delta[timeStep] = (error[t] / sumPost);
                                }
                            }
                        }
                } else {
                    throw new NeuralNetworkException("Could not do a CharacterSequence forward pass on RecurrentNeuralNetwork because lossFunction was unknown or invalid: " + lossFunction);
                }

        } else if (sequence instanceof TimeSeries) {
            TimeSeries series = (TimeSeries)sequence;
            
             if (lossFunction == LossFunction.NONE) {
                //for no loss function we can just return the sum of the outputs overall time steps, and can set each output node's delta to 1
                for (int timeStep = 0; timeStep < series.getLength() - 1; timeStep++) {
                    for (int number = 0; number < nOutputs; number++) {
                        RecurrentNode outputNode = layers[outputLayer][number];
                        lossSum += outputNode.postActivationValue[timeStep];
                        outputNode.delta[timeStep] = 1.0;
                    }
                }

            }else if (lossFunction == LossFunction.L1_NORM) {
                //TODO: Implement this for Programming Assignment 2 - Part 3
                
                for (int timeStep = 0; timeStep < series.getLength() -1; timeStep++){
                    for (int number = 0; number < nOutputs; number++){
                        //expected value will be timeStep + 1
                        double expectedTarget = series.getOutputValue(timeStep + 1, number);
                        RecurrentNode outputNode = layers[outputLayer][number];
                        
                        lossSum += Math.abs(outputNode.postActivationValue[timeStep] - expectedTarget); 
                        //update delta abs(yi-zi)/(yi-zi)
                        outputNode.delta[timeStep] = Math.abs(outputNode.postActivationValue[timeStep] - expectedTarget) / (outputNode.postActivationValue[timeStep] - expectedTarget);
                        }
                    } 
            } else if (lossFunction == LossFunction.L2_NORM) {
                //TODO: Implement this for Programming Assignment 2 - Part 3
                for (int timeStep = 0; timeStep < series.getLength() -1; timeStep++){
                    
                    double l2 = 0.0;
                    double[] error = new double[nOutputs];
                    
                    for (int number = 0; number < nOutputs; number++){
                        //expected value will be timeStep + 1
                        double expectedTarget = series.getOutputValue(timeStep + 1, number);
                        RecurrentNode outputNode = layers[outputLayer][number];
                        
                        error[number] = (outputNode.postActivationValue[timeStep] - expectedTarget); 
                        l2 += error[number] * error[number];
                        }
                    
                    //sum up the lossSum
                    lossSum += Math.sqrt(l2); 
                    
                    //update delta abs(yi-zi)/(yi-zi)
                    for (int i = 0; i< nOutputs; i++){
                        RecurrentNode outputNode = layers[outputLayer][i];
                        outputNode.delta[timeStep] = error[i]/Math.sqrt(l2);
                        }
                    }

            } else {
                throw new NeuralNetworkException("Could not do a TimeSeries forward pass on RecurrentNeuralNetwork because lossFunction was unknown: " + lossFunction);
            }
        }

        return lossSum;
    }

    /**
     * This performs multiple forward passes through the neural network
     * by multiple instances are returns the output sum.
     *
     * @param sequences is the set of CharacterSequences to pass through the network
     *
     * @return the sum of their outputs
     */
    public double forwardPass(List<Sequence> sequences) throws NeuralNetworkException {
        double sum = 0.0;

        for (Sequence sequence : sequences) {
            sum += forwardPass(sequence);
        }

        return sum;
    }

    /**
     * This performs multiple forward passes through the recurrent neural network
     * and calculates how many of the character predictions for every
     * sequence were classified correctly.
     *
     * @param sequences is the set of CharacterSequences to pass through the network
     *
     * @return a percentage (between 0 and 1) of how many character predictions were
     * correctly classified
     */
    public double calculateAccuracy(List<CharacterSequence> sequences) throws NeuralNetworkException {
        //TODO: need to implement this for Programming Assignment 2 - Part 2
        //the output node with the maximum value is the predicted class
        //you need to sum up how many of these match the actual class
        //for each time step of each sequence, and then calculate: 
        //num correct / total
        //to get a percentage accuracy
        
        //total training number
        int total = 0;
        int correct = 0;

        for (CharacterSequence characterSequence: sequences){
            
            total += characterSequence.getLength() - 1;
            for (int timeStep = 0; timeStep < characterSequence.getLength() -1; timeStep ++){
                
                int position = (int) characterSequence.valueAt(timeStep + 1);
                this.forwardPass(characterSequence);
                double output[][] = this.getOutputValues(characterSequence);
                int predictedClass = 0;
                
                //find the largest output value
                double maxValue = output[0][0];
                for (int i=0; i < output[timeStep].length; i++){
                    
                    if (output[timeStep][i] > maxValue){
                        predictedClass = i;
                        maxValue = output[timeStep][i];
                        }
                    }
                if (position == predictedClass){
                    correct++;
                    }

                }
            }

            return correct * (1.0/total);

        //throw new NeuralNetworkException("RecurrentNeuralNetwork.calculateAccuracy not implemented!");
    }


    /**
     * This gets the output values of the neural network 
     * after a forward pass, this will be a 2 dimensional array for
     * an RNN because we'll have outputs for each time step
     *
     * @param seuence is the CharacterSequence which generated the the output values, we need this to get the length of the sequence
     *
     * @return a two dimensional array of the output values from this neural network for each time step
     */
    public double[][] getOutputValues(Sequence sequence) {
        //the number of output values is the number of output nodes
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        double[][] outputValues = new double[sequence.getLength() - 1][nOutputs];

        //we will have 1 less output because we're predicting one character ahead
        for (int timeStep = 0; timeStep < sequence.getLength() - 1; timeStep++) {
            for (int number = 0; number < nOutputs; number++) {
                outputValues[timeStep][number] = layers[outputLayer][number].postActivationValue[timeStep];
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
     */
    public double[] getNumericGradient(Sequence sequence) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        double originalWeights[] = this.getWeights();
        double[] gradient = new double[this.numberWeights];
        double tempWeights[] = this.getWeights();
        
        for (int i=0; i< this.numberWeights; i++){
            double output1 = 0.0;
            double output2 = 0.0;

            
            tempWeights[i] = originalWeights[i] + H;
            this.setWeights(tempWeights);
            output1 = this.forwardPass(sequence);

            tempWeights[i] = originalWeights[i] - H;
            this.setWeights(tempWeights);
            output2 = this.forwardPass(sequence);

            gradient[i] = (output1 - output2)/ (2*H);
            tempWeights[i] = originalWeights[i];
            }
        
        this.setWeights(originalWeights);
        
        return gradient;
        //throw new NeuralNetworkException("RecurrentNeuralNetwork.getNumericGradient(Sequence sequence) not implemented!");
    }

    /**
     * This calculates the gradient of the neural network with it's current
     * weights for a given DataSet Instance using the finite difference method:
     * gradient[i] = (f(x where x[i] = x[i] + H) - f(x where x[i] = x[i] - H)) / 2h
     */
    public double[] getNumericGradient(List<Sequence> sequences) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        double[] gradient = new double[this.numberWeights];

        for (Sequence sequence: sequences){
            double tempGradient[] = this.getNumericGradient(sequence);

            for (int i=0; i < this.numberWeights; i++){
                gradient[i] += tempGradient[i];
                }    
            }

        return gradient;
        //throw new NeuralNetworkException("RecurrentNeuralNetwork.getNumericGradient(List<Sequence> sequences) not implemented!");
    }


    /**
     * This performs a backward pass through the neural network given 
     * outputs from the given instance. This will set the deltas in
     * all the edges and nodes which will be used to calculate the 
     * gradient and perform backpropagation.
     *
     */
    public void backwardPass(Sequence sequence) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        //note you should start at sequence.getLength() - 2 (not -1) as this is the last
        //time step that was passed forward through the RNN
        
        int OutputLayer = layers.length -1;
        //set input values differently for time series and character sequences
        if (sequence instanceof CharacterSequence) {
            CharacterSequence characterSequence = (CharacterSequence)sequence;

            //set the input nodes for each time step in the CharacterSequence
            for (int timeStep = characterSequence.getLength() -2 ; timeStep >= 0; timeStep--) {
                for (int number = OutputLayer; number >= 0 ; number--) {
                    for (RecurrentNode outputNode: layers[number]){
                        outputNode.propagateBackward(timeStep);
                        }
                    }
                }

        } else if (sequence instanceof TimeSeries) {
            TimeSeries series = (TimeSeries)sequence;

            //set the input nodes for each time step in the TimeSeries
            for (int timeStep = series.getLength() - 2; timeStep >= 0; timeStep--) {
                for (int number = OutputLayer; number >= 0; number--) {
                    for (RecurrentNode outputNode : layers[number]){
                        outputNode.propagateBackward(timeStep);
                        }
                    }
                }

            }
        }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the RecurrentNeuralNetwork.backwardPass(Sequence)) Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient).
     *
     * @param sequence is the training CharacterSequence for the forward and 
     * backward pass.
     */
    public double[] getGradient(Sequence sequence) throws NeuralNetworkException {
        forwardPass(sequence);
        backwardPass(sequence);

        return getDeltas();
    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the RecurrentNeuralNetwork.backwardPass(Instance))* Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient). The resulting gradient should be the sum of
     * each delta for each instance.
     *
     * @param sequences are the training CharacterSequences for the forward and 
     * backward passes.
     */
    public double[] getGradient(List<Sequence> sequences) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        double[] gradient = new double[this.numberWeights];
        
        for (Sequence sequence: sequences){
            //get the gradient through forward,backward pass.
            forwardPass(sequence);
            backwardPass(sequence);
            
            //sum the gradient of each sequence
            double temp_gradient[] = this.getDeltas();
            for (int i=0; i < gradient.length; i++){
                gradient[i] += temp_gradient[i];
                }
            }
        
        return gradient;
        //throw new NeuralNetworkException("RecurrentNeuralNetwork.getGradient(List<Sequence> sequences) not implemented!");
    }
}
