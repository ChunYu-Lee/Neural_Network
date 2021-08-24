package network;

import java.util.List;

import java.util.Arrays;

import data.Instance;

import util.Log;


public class NeuralNetwork {
    //this is the loss function for the output of the neural 
    //network, you will use this in PA1-3
    LossFunction lossFunction;
    

    //this is the total number of weights in the neural network
    int numberWeights;
    
    //layers contains all the nodes in the neural network
     Node[][] layers;

    public NeuralNetwork(int inputLayerSize, int[] hiddenLayerSizes, int outputLayerSize, LossFunction lossFunction) {
        this.lossFunction = lossFunction;

        //the number of layers in the neural network is 2 plus the number of hidden layers,
        //one additional for the input, and one additional for the output.

        //create the outer array of the 2-dimensional array of nodes
        layers = new Node[hiddenLayerSizes.length + 2][];
        
        //we will progressively calculate the number of weights as we create the network. the
        //number of edges will be equal to the number of hidden nodes (each has a bias weight, but
        //the input and output nodes do not) plus the number of edges
        numberWeights = 0;

        Log.info("creating a neural network with " + hiddenLayerSizes.length + " hidden layers.");
        for (int layer = 0; layer < layers.length; layer++) {
            
            //determine the layer size depending on the layer number, 0 is the
            //input layer, and the last layer is the output layer, all others
            //are hidden layers
            int layerSize;
            NodeType nodeType;
            ActivationType activationType;
            if (layer == 0) {
                //this is the input layer
                layerSize = inputLayerSize;
                nodeType = NodeType.INPUT;
                activationType = ActivationType.LINEAR;
                Log.info("input layer " + layer + " has " + layerSize + " nodes.");

            } else if (layer < layers.length - 1) {
                //this is a hidden layer
                layerSize = hiddenLayerSizes[layer - 1];
                nodeType = NodeType.HIDDEN;
                activationType = ActivationType.TANH;
                Log.info("hidden layer " + layer + " has " + layerSize + " nodes.");

                //increment the number of weights by the number of nodes in
                //this hidden layer
                numberWeights += layerSize; 
            } else {
                //this is the output layer
                layerSize = outputLayerSize;
                nodeType = NodeType.OUTPUT;
                activationType = ActivationType.SIGMOID;
                Log.info("output layer " + layer + " has " + layerSize + " nodes.");
            }

            //create the layer with the right length and right node types
            layers[layer] = new Node[layerSize];
            for (int j = 0; j < layers[layer].length; j++) {
                layers[layer][j] = new Node(layer, j /*i is the node number*/, nodeType, activationType);
            }
        }
    }

    /**
     * This gets the number of weights in the NeuralNetwork, which should
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
     * This returns an array of every weight (including biases) in the NeuralNetwork.
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
                    throw new NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + numberWeights + ") but when getting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return weights;
    }

    /**
     * This sets every weight (including biases) in the NeuralNetwork, it sets them in
     * the same order that they are retreived by the getWeights method.
     * This will be very useful in backpropagation and sanity checking. 
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public void setWeights(double[] newWeights) throws NeuralNetworkException {
        if (numberWeights != newWeights.length) {
            throw new NeuralNetworkException("Could not setWeights because the number of new weights: " + newWeights.length + " was not equal to the number of weights in the NeuralNetwork: " + numberWeights);
        }

        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nWeights = layers[layer][nodeNumber].setWeights(position, newWeights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + numberWeights + ") but when setting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }
    }

    /**
     * This returns an array of every weight (including biases) in the NeuralNetwork.
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
                    throw new NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + numberWeights + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return deltas;
    }


    /**
     * This adds edges to the NeuralNetwork, connecting each node
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
                    Node inputNode = layers[layer][inputNodeNumber];
                    Node outputNode = layers[layer + 1][outputNodeNumber];
                    new Edge(inputNode, outputNode);

                    //as we added an edge, the number of weights should increase by 1
                    numberWeights++;
                    Log.trace("numberWeights now: " + numberWeights);
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

        //TODO: Complete this function for Programming Assignment 1 - Part 1. BONUS: allow it to it create edges that can skip layers
	    //Find out the Node inside the layers
	    Node inputNode = layers[inputLayer][inputNumber];
	    Node outputNode = layers[outputLayer][outputNumber];
	    new Edge(inputNode, outputNode);

	    //as we added an edge, the number of weights should increase by 1
	    numberWeights++;
	    Log.trace("numberWeights now: " + numberWeights);

    }

    /**
     * This initializes the weights properly by setting the incoming
     * weights for each edge using a random normal distribution (i.e.,
     * a gaussian distribution) and dividing the randomly generated
     * weight by sqrt(n) where n is the fan-in of the node. It also
     * sets the bias for each node to the given parameter.
     *
     * For example, if we have a node N which has 5 input edges,
     * the weights of each of those edges will be generated by
     * Random.nextGaussian()/sqrt(5). The best way to do this is
     * to iterate over each node and have it use the 
     * Node.initializeWeightsAndBias(double bias) method.
     *
     * @param bias is the value to set the bias of each node to.
     */
    public void initializeRandomly(double bias) {
        //TODO: You need to implement this for PA1-3

        for(int i = 1; i < layers.length; i++){
            for(int j = 0; j < layers[i].length; j++){
                Node node = layers[i][j];
                if (node.nodeType == NodeType.OUTPUT){
                    //the output layer usually does not have bias.
                    node.initializeWeightsAndBias(0.0);
                }else{
                    node.initializeWeightsAndBias(bias);
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
    public double forwardPass(Instance instance) throws NeuralNetworkException {
        //be sure to reset before doing a forward pass
        reset();

        //TODO: You need to implement this for Programming Assignment 1 - Part 1
	//set the input node value
	for (int i = 0; i < layers[0].length; i++){
		layers[0][i].preActivationValue = instance.inputs[i];
	}
	
	//forward pass
	for(int j = 0; j < layers.length; j++){
		for(int i =0; i < layers[j].length; i++){
			layers[j][i].propagateForward();
	   }
	}
	

        //The following is needed for PA1-3 and PA1-4
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        double outputSum = 0;
        if (lossFunction == LossFunction.NONE) {

            //just sum up the outputs
            for (int number = 0; number < nOutputs; number++) {
                Node outputNode = layers[outputLayer][number];
                outputNode.delta = 1;
                outputSum += outputNode.postActivationValue;
                }

        } else if (lossFunction == LossFunction.L1_NORM) {
            //TODO: Implement this for PA1-3 

            for (int number = 0; number < nOutputs; number++) {
                
                Node outputNode = layers[outputLayer][number];
                outputNode.delta = Math.abs(outputNode.postActivationValue - instance.expectedOutputs[number])/(outputNode.postActivationValue - instance.expectedOutputs[number]);
                outputNode.postActivationValue = Math.abs(outputNode.postActivationValue - instance.expectedOutputs[number]);
                outputSum += outputNode.postActivationValue;
                }
        } else if (lossFunction == LossFunction.L2_NORM) {
            //TODO: Implement this for PA1-3

            double error[] = new double[nOutputs];

            //calculate the loss value
            for (int number = 0; number < nOutputs; number++) {
                
                Node outputNode = layers[outputLayer][number];
                //save the zi-yi before the postActivationValue being updated and use the value to update the delta.
                //you can choose to update or leave the postActivationValue.
                error[number] = outputNode.postActivationValue - instance.expectedOutputs[number];
                outputNode.postActivationValue = error[number] * error[number];
                outputSum += outputNode.postActivationValue;
                outputNode.postActivationValue = Math.sqrt(outputNode.postActivationValue);
                }

            //sqrt the outputSum
            outputSum = Math.sqrt(outputSum);
            
            //update the delta
            for (int i = 0; i < nOutputs; i++){
                Node outputNode = layers[outputLayer][i];
                outputNode.delta = error[i] / outputSum;
                }

        }else if (lossFunction == LossFunction.SVM) {
            //TODO: Implement this for PA1-4
            double error[] = new double[nOutputs];
            int position = (int) instance.expectedOutputs[0];
            int yDelta = 0;

            //calculate the error
            for(int i=0; i<nOutputs; i++){
               
                Node outputNode = layers[outputLayer][i];

                if(i == position){
                    
                    error[i] = 0.0;
                }else{
                    
                    error[i] = Math.max(0, outputNode.postActivationValue - layers[outputLayer][position].postActivationValue + 1) ; 
                    }
                } 
            
            //sum all the error
            for (double number: error){
                outputSum += number;
                if (number >0){
                    yDelta++;
                    }
                }

            //set all the delta
            for(int j=0; j<nOutputs; j++){
                Node outputNode = layers[outputLayer][j];
                
                if(j == position){

                    outputNode.delta = -yDelta;
                }else{
                    if (error[j] >0){
                        outputNode.delta = 1;
                    }else{
                        outputNode.delta = 0;
                        }
                    
                    }
            }

        } else if (lossFunction == LossFunction.SOFTMAX) {
            //TODO: Implement this for PA1-4
            
            
            double error[] = new double[nOutputs];
            double sumPost = 0.0;
            int position = (int) instance.expectedOutputs[0];
            double maxValue = layers[outputLayer][0].postActivationValue;

            //get the max value and position
            for (int number = 0; number < nOutputs; number++) {
                
                Node outputNode = layers[outputLayer][number]; 
                if (outputNode.postActivationValue > maxValue){
                    maxValue = outputNode.postActivationValue;
                    } 

                }
            //calculate the error and sum all the ezi
            for (int n =0; n < nOutputs; n++){
                
                Node outputNode = layers[outputLayer][n];

                error[n] = Math.exp(outputNode.postActivationValue - maxValue);
                sumPost += error[n];
                }
            
            //update the delta and sum the loss value 
            for (int t=0; t < nOutputs; t++){
                
                Node outputNode = layers[outputLayer][t];
                
                if (t == position){
                    outputNode.delta =  (error[t] / sumPost) -1;
                    outputSum = -Math.log(error[t] / sumPost);
                }else{
                    outputNode.delta = (error[t] / sumPost);
                    }
                }
            
        } else {
            throw new NeuralNetworkException("Could not do forward pass on NeuralNetwork because lossFunction was unknown: " + lossFunction);
        } 

        return outputSum;
    }

    /**
     * This performs multiple forward passes through the neural network
     * by multiple instances are returns the output sum.
     *
     * @param instances is the set of instances to pass through the network
     *
     * @return the sum of their outputs
     */
    public double forwardPass(List<Instance> instances) throws NeuralNetworkException {
        double sum = 0.0;

        for (Instance instance : instances) {
            sum += forwardPass(instance);
        }

        return sum;
    }

    /**
     * This performs multiple forward passes through the neural network
     * and calculates how many of the instances were classified correctly.
     *
     * @param instances is the set of instances to pass through the network
     *
     * @return a percentage (between 0 and 1) of how many instances were
     * correctly classified
     */
    public double calculateAccuracy(List<Instance> instances) throws NeuralNetworkException {
        //TODO: need to implement this for PA1-4
        //the output node with the maximum value is the predicted class
        //you need to sum up how many of these match the actual class
        //from the instance, and then calculate: num correct / total
        //to get a percentage accuracy
        
        int total = instances.size();
        int correct = 0;
        for(Instance instance: instances){
            //do forward pass, and get the outputs
            this.forwardPass(instance);
            int position = (int) instance.expectedOutputs[0];
            double[] output = this.getOutputValues();
            int predictedClass = 0;
            double maxValue = output[0];

            //get the predicted class
            for (int i=0; i < output.length; i++){
                
                if ( output[i] > maxValue){
                    predictedClass = i;
                    maxValue = output[i];
                    }
                }
            if (position == predictedClass){
                correct++;
                }
            } 
        return (correct * 1.0/total);
        //throw new NeuralNetworkException("calculateAccuracy(List<Instance> instances) was not yet implemented!");
    }


    /**
     * This gets the output values of the neural network 
     * after a forward pass.
     *
     * @return an array of the output values from this neural network
     */
    public double[] getOutputValues() {
        //the number of output values is the number of output nodes
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        double[] outputValues = new double[nOutputs];

        for (int number = 0; number < nOutputs; number++) {
            outputValues[number] = layers[outputLayer][number].postActivationValue;
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
    public double[] getNumericGradient(Instance instance) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 1 - Part 2
    
	//keep the original weights	
	double[] originalWeights = this.getWeights();

	//build up the temp_weights by copy the getWeights
	double[] tempWeights = this.getWeights();
		
	//for loop- add H and deduct H to every weight and run the forward pass
	double[] gradient = new double[this.getNumberWeights()];
	for (int k=0; k < gradient.length; k++){
		double output1;
		double output2;
		
		//add H to weight
		tempWeights[k] = originalWeights[k] + H;
		this.setWeights(tempWeights);
		output1 = forwardPass(instance);
		
		//deduct H to weight
		tempWeights[k] = originalWeights[k] - H;
		this.setWeights(tempWeights);
		output2 = forwardPass(instance);

		gradient[k] =  (output1 - output2)/(2.0*H);
		tempWeights[k] = originalWeights[k];
	}
	
	//resume the weights in NN
	this.setWeights(originalWeights);

	return gradient;
        
	//throw new NeuralNetworkException("getNumericGradient(Instance instance) was not yet implemented!");
    }

    /**
     * This calculates the gradient of the neural network with it's current
     * weights for a given DataSet Instance using the finite difference method:
     * gradient[i] = (f(x where x[i] = x[i] + H) - f(x where x[i] = x[i] - H)) / 2h
     */
    public double[] getNumericGradient(List<Instance> instances) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 1 - Part 2
	
	//set up the return gradient
	double[] gradient = new double[this.getNumberWeights()];

	//for loop List<Instance> and build up the gradient
	for(Instance inst: instances){
		double tempGradient[] = this.getNumericGradient(inst);
		for (int i=0; i < tempGradient.length; i++){
			gradient[i] += tempGradient[i];
		
            if (i > gradient.length) {
                throw new NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + numberWeights + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
	    }

	return gradient;
	 
    }


    /**
     * This performs a backward pass through the neural network given 
     * outputs from the given instance. This will set the deltas in
     * all the edges and nodes which will be used to calculate the 
     * gradient and perform backpropagation.
     *
     */
    public void backwardPass(Instance instance) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 1 - Part 2
          
        //loop the whole layers 
        for(int num = layers.length; num >0; num--){
            for(Node node : layers[num - 1]){
                node.propagateBackward();
                }
            }
           
        //    throw new NeuralNetworkException("The instance was passed forward is not same as this one");  
         
    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the NeuralNetwork.backwardPass(Instance))* Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient).
     *
     * @param instance is the training instance/sample for the forward and 
     * backward pass.
     */
    public double[] getGradient(Instance instance) throws NeuralNetworkException {
        forwardPass(instance);
        backwardPass(instance);
        
        return getDeltas();
    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the NeuralNetwork.backwardPass(Instance))* Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient). The resulting gradient should be the sum of
     * each delta for each instance.
     *
     * @param instances are the training instances/samples for the forward and 
     * backward passes.
     */
    public double[] getGradient(List<Instance> instances) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 1 - Part 2
	
	    //set up Gradient
	    double[] Gradient = new double[this.getNumberWeights()];

	    //for loop the list of instances
	    for (Instance inst: instances){
		    forwardPass(inst);
		    backwardPass(inst);
		    
		//acquire the vale from Deltas
		    double tempGradient[] = this.getDeltas();
		    for(int i=0; i < tempGradient.length; i++){
			    Gradient[i] += tempGradient[i];

                if (i > Gradient.length) {
                    throw new NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + numberWeights + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                    }
	            }
	        }

	return Gradient;
        //throw new NeuralNetworkException("backwardPass(List<Instance> instances) was not yet implemented!");
    
    }
}
