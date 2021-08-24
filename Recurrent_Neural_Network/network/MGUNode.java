/**
 * This class represents a Node in the neural network. It will
 * have a list of all input and output edges, as well as its
 * own value. It will also track it's layer in the network and
 * if it is an input, hidden or output node.
 */
package network;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import util.Log;

public class MGUNode extends RecurrentNode {

    //these are the weight values(6) for the MGU node
    double hw;
    double fu;
    double fw;
    double hu;

    //these are the bias values(1) for the MGU node
    double fb;
    double hb;

    //these are the deltas for the weights and biases
    public double delta_hw;
    public double delta_fu;
    public double delta_fw;
    public double delta_hu;

    public double delta_fb;
    public double delta_hb;
    
    //ht is the postActivationValue
    //this is the delta value for ht in the diagram, it will be
    //set to the sum of the delta coming in from the outputs (delta)
    //plus whatever deltas came in from
    //the subsequent time step during backprop
    public double[] delta_ht;

    //f values for each time step
    public double[] f;

    //forget gate values for each time step
    public double[] h;



    /**
     * This creates a new node at a given layer in the
     * network and specifies it's type (either input,
     * hidden, our output).
     *
     * @param layer is the layer of the Node in
     * the neural network
     * @param type is the type of node, specified by
     * the Node.NodeType enumeration.
     */
    public MGUNode(int layer, int number, NodeType nodeType, int maxSequenceLength) {
        super(layer, number, nodeType, maxSequenceLength, null);

        delta_ht = new double[maxSequenceLength];

        f = new double[maxSequenceLength];
        h = new double[maxSequenceLength];

   }

    /**
     * This resets the values which need to be recalcualted for
     * each forward and backward pass. It will also reset the
     * deltas for outgoing nodes.
     */
    public void reset() {
        //use RecurrentNode's reset to reset everything this has inherited from
        //RecurrentNode, then reset the MGUNode's fields
        super.reset();
        Log.trace("Resetting MGU node: " + toString());

        for (int timeStep = 0; timeStep < maxSequenceLength; timeStep++) {
            f[timeStep] = 0;
            h[timeStep] = 0;
            delta_ht[timeStep] = 0;
        }

        delta_hw = 0;
        delta_fu = 0;
        delta_fw = 0;
        delta_hu = 0;
        
        delta_fb = 0;
        delta_hb = 0;
    }


    /**
     * We need to override the getWeightNames from RecurrentNode as
     * an MGUNode will have 6 weight and bias names as opposed to
     * just one bias.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weightNames is the array of weight nameswe're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeightNames(int position, String[] weightNames) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            weightNames[position] = "MGU Node [layer " + layer + ", number " + number + ", hw]";
            weightNames[position + 1] = "MGU Node [Layer " + layer + ", number " + number + ", fu]";

            weightNames[position + 2] = "MGU Node [Layer " + layer + ", number " + number + ", fw]";
            weightNames[position + 3] = "MGU Node [Layer " + layer + ", number " + number + ", hu]";

            weightNames[position + 4] = "MGU Node [Layer " + layer + ", number " + number + ", fb]";
            weightNames[position + 5] = "MGU Node [Layer " + layer + ", number " + number + ", hb]";

            weightCount += 6;
        }

        for (Edge edge : outputEdges) {
            String targetType = "";
            if (edge.outputNode instanceof MGUNode) targetType = "MGU ";
            weightNames[position + weightCount] = "Edge from MGU Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + edge.outputNode.layer + ", number " + edge.outputNode.number + "]";
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            String targetType = "";
            if (recurrentEdge.outputNode instanceof MGUNode) targetType = "MGU ";

            weightNames[position + weightCount] = "Recurrent Edge from MGU Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + recurrentEdge.outputNode.layer + ", number " + recurrentEdge.outputNode.number + "]";
            weightCount++;
        }


        return weightCount;
    }



    /**
     * We need to override the getWeights from RecurrentNode as
     * an MGUNode will have 6 weights and biases as opposed to
     * just one bias.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weights is the array of weights we're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeights(int position, double[] weights) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            weights[position] = hw;
            weights[position + 1] = fu;
            weights[position + 2] = fw;
            weights[position + 3] = hu;
            
            weights[position + 4] = fb;
            weights[position + 5] = hb;

            weightCount += 6;
        }

        for (Edge edge : outputEdges) {
            weights[position + weightCount] = edge.weight;
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            weights[position + weightCount] = recurrentEdge.weight;
            weightCount++;
        }

        return weightCount;
    }

    /**
     * We need to override the getDeltas from RecurrentNode as
     * an MGUNode will have 6 weights and biases as opposed to
     * just one bias.
     *
     * @param position is the index to start setting deltas in the deltas parameter
     * @param deltas is the array of deltas we're setting.
     *
     * @return the number of deltas set in the deltas parameter
     */
    public int getDeltas(int position, double[] deltas) {
        int deltaCount = 0;

        //the first delta set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            deltas[position] = delta_hw;
            deltas[position + 1] = delta_fu;
            deltas[position + 2] = delta_fw;
            deltas[position + 3] = delta_hu;
            
            deltas[position + 4] = delta_fb;
            deltas[position + 5] = delta_hb;

            deltaCount += 6;
        }

        for (Edge edge : outputEdges) {
            deltas[position + deltaCount] = edge.weightDelta;
            deltaCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            deltas[position + deltaCount] = recurrentEdge.weightDelta;
            deltaCount++;
        }

        return deltaCount;
    }


    /**
     * We need to override the getDeltas from RecurrentNode as
     * an MGUNode will have 6 weights and biases as opposed to
     * just one bias.
     *
     * @param position is the starting position in the weights parameter to start
     * setting weights from.
     * @param weights is the array of weights we are setting from
     *
     * @return the number of weights gotten from the weights parameter
     */

    public int setWeights(int position, double[] weights) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            hw = weights[position];
            fu = weights[position + 1];
            fw = weights[position + 2];
            hu = weights[position + 3];
            
            fb = weights[position + 4];
            hb = weights[position + 5];

            weightCount += 6;
        }

        for (Edge edge : outputEdges) {
            edge.weight = weights[position + weightCount];
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            recurrentEdge.weight = weights[position + weightCount];
            weightCount++;
        }

        return weightCount;
    }

    double sigmoid(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }

    /**
     * This propagates the postActivationValue at this MGU node
     * to all it's output nodes.
     */
    public void propagateForward(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 4
        //NOTE: recurrent edges need to be propagated forward from this timeStep to
        //their targetNode at timeStep + the recurrentEdge's timeSkip

        //calculate all the values. input = this.preActivationValue, output(ht) = postActivationValue
        if (timeStep == 0){
            this.f[timeStep] = sigmoid(fw*this.preActivationValue[timeStep] + fu * 0 + fb);
            this.h[timeStep] = Math.tanh(hw*this.preActivationValue[timeStep] + f[timeStep] * hu * 0 + hb);
            //ht
            this.postActivationValue[timeStep] = (1 - f[timeStep]) * 0 + (f[timeStep] * h[timeStep]);
        }else{
            this.f[timeStep] = sigmoid(fw*this.preActivationValue[timeStep] + fu * this.postActivationValue[timeStep -1] + fb);
            this.h[timeStep] = Math.tanh(hw*this.preActivationValue[timeStep] + f[timeStep] * hu * this.postActivationValue[timeStep -1] + hb);
            //ht
            this.postActivationValue[timeStep] = (1 - f[timeStep]) * this.postActivationValue[timeStep -1] + (f[timeStep] * h[timeStep]);
        }


        //propagate the edge forward
        for (Edge edge : outputEdges){
            edge.outputNode.preActivationValue[timeStep] += edge.weight * this.postActivationValue[timeStep];
            }
        //propagate the recurrent edge forward
        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            MGUNode node = (MGUNode)recurrentEdge.outputNode;
            node.preActivationValue[timeStep + recurrentEdge.timeSkip] += recurrentEdge.weight * this.postActivationValue[timeStep];
        }
    }

    /**
     * This propagates the delta back from this node
     * to its incoming edges.
     */
    public void propagateBackward(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 4
        //node that delta[timeStep] is the delta coming in for the output (delta_ht in the slides)

        //variable
        double e = 0.0;
        double g = 0.0;
        double i = 0.0;
        //delta ht
        double d_ht = 0.0;

        //ht-1
        double number_ht = 0.0;

        //last timeStep
        if (timeStep == maxSequenceLength - 1){
            d_ht = 0 + this.delta[timeStep];
        }else{
            d_ht = delta_ht[timeStep] + this.delta[timeStep];
        }

        if (timeStep > 0){
            number_ht = this.postActivationValue[timeStep - 1];
        }

        //variable update
        e =  f[timeStep] * d_ht;
        g = e * (1 - h[timeStep] * h[timeStep]);
        i = ((h[timeStep] - number_ht) *d_ht + number_ht * hu * g) * ((1 - f[timeStep]) * f[timeStep]);

        //bias deltas
        delta_fb += i;
        delta_hb += g;

        //weight deltas
        delta_hw += g * this.preActivationValue[timeStep];
        delta_fu += i * number_ht;
        delta_fw += i * this.preActivationValue[timeStep];
        delta_hu += f[timeStep] * number_ht * g;


        //input and previous output deltas
        if (timeStep >0){
            delta_ht[timeStep -1] = (fu * i) + ((1 - f[timeStep]) * d_ht) + (f[timeStep] * hu * g);
            }

        this.delta[timeStep] = (g * hw) + (i * fw);


        //propagate the edge backward
        for (Edge edge : inputEdges){
            edge.propagateBackward(timeStep, this.delta[timeStep]);
            }
        //propagate the recurrent edge backward
        //set the ct[timeStep + timeSkip] = ct[timeStep], then in the future timeStep just use ct[timeStep] as ct(t-1)
        for (RecurrentEdge recurrentEdge : inputRecurrentEdges) {
            recurrentEdge.propagateBackward(timeStep, this.delta[timeStep]);
            }
    }

    /**
     *  This sets the node's bias to the bias parameter and then
     *  randomly initializes each incoming edge weight by using
     *  Random.nextGaussian() / sqrt(N) where N is the number
     *  of incoming edges.
     *
     *  @param bias is the bias to initialize this node's bias to
     */
    public void initializeWeightsAndBiasKaiming(int fanIn, double bias) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2

        Random rand = new Random();
        this.bias = bias;

        //set up the weight and bias in MGU node

        hw = rand.nextGaussian();
        fu = rand.nextGaussian();
        fw = rand.nextGaussian();
        hu = rand.nextGaussian();

        // bias values for the MGU node
        fb = rand.nextGaussian() - 1;
        hb = rand.nextGaussian();

        //set up the weight in the edges
        for (Edge edge : this.inputEdges){
             edge.weight = rand.nextGaussian() * Math.sqrt(2/fanIn);
             }
        //set up the weight in the recurrentEdges
        for (RecurrentEdge recurrentEdge: this.inputRecurrentEdges){
             recurrentEdge.weight = rand.nextGaussian() * Math.sqrt(2/fanIn);
             }
    }

    /**
     *  This sets the node's bias to the bias parameter and then
     *  randomly intializes each incoming edge weight uniformly
     *  at random (you can use Random.nextDouble()) between
     *  +/- sqrt(6) / sqrt(fan_in + fan_out)
     *
     *  @param bias is the bias to initialize this node's bias to
     */
    public void initializeWeightsAndBiasXavier(int fanIn, int fanOut, double bias) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        this.bias = bias;
        Random rand = new Random();

        hw = rand.nextGaussian();
        fu = rand.nextGaussian();
        fw = rand.nextGaussian();
        hu = rand.nextGaussian();

        // bias values for the MGU node
        fb = rand.nextGaussian() - 1;
        hb = rand.nextGaussian();

        //set up weight in the edges
        for (Edge edge: this.inputEdges){
            //Get the number from -1 to 1(exclusive--not sure how to solve it yet). Then multiply sqrt(6/(fanIn+fanOut))
            edge.weight = (2 * rand.nextDouble() -1) * (Math.sqrt(6/(fanIn + fanOut)));
            }

        //set up the weight in the recurrent edges
        for (RecurrentEdge recurrentEdge: this.inputRecurrentEdges){
            //Get the number from -1 to 1
            recurrentEdge.weight = (2 * rand.nextDouble() -1) * (Math.sqrt(6/(fanIn + fanOut)));
            }
    }


    /**
     * Prints concise information about this node.
     *
     * @return The node as a short string.
     */
    public String toString() {
        return "[MGU Node - layer: " + layer + ", number: " + number + ", type: " + nodeType + "]";
    }
}
