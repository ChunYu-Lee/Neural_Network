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

public class UGRNNNode extends RecurrentNode {

    //these are the weight values(6) for the UGRNN node
    double gh;
    double ch;
    double gw;
    double cw;

    //these are the bias values(1) for the UGRNN node
    double gb;
    double cb;

    //these are the deltas for the weights and biases
    public double delta_gh;
    public double delta_ch;
    public double delta_gw;
    public double delta_cw;

    public double delta_gb;
    public double delta_cb;

    //ht is the postActivationValue
    //this is the delta value for ht in the diagram, it will be
    //set to the sum of the delta coming in from the outputs (delta)
    //plus whatever deltas came in from
    //the subsequent time step during backprop
    public double[] delta_ht;

    //g values for each time step
    public double[] g;

    //C values for each time step
    public double[] C;



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
    public UGRNNNode(int layer, int number, NodeType nodeType, int maxSequenceLength) {
        super(layer, number, nodeType, maxSequenceLength, null);

        delta_ht = new double[maxSequenceLength];

        g = new double[maxSequenceLength];
        C = new double[maxSequenceLength];

   }

    /**
     * This resets the values which need to be recalcualted for
     * each forward and backward pass. It will also reset the
     * deltas for outgoing nodes.
     */
    public void reset() {
        //use RecurrentNode's reset to reset everything this has inherited from
        //RecurrentNode, then reset the UGRNNNode's fields
        super.reset();
        Log.trace("Resetting UGRNN node: " + toString());

        for (int timeStep = 0; timeStep < maxSequenceLength; timeStep++) {
            g[timeStep] = 0;
            C[timeStep] = 0;
            delta_ht[timeStep] = 0;
        }

        delta_gh = 0;
        delta_ch = 0;
        delta_gw = 0;
        delta_cw = 0;

        delta_gb = 0;
        delta_cb = 0;
    }


    /**
     * We need to override the getWeightNames from RecurrentNode as
     * an UGRNNNode will have 6 weight and bias names as opposed to
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
            weightNames[position] = "UGRNN Node [layer " + layer + ", number " + number + ", gh]";
            weightNames[position + 1] = "UGRNN Node [Layer " + layer + ", number " + number + ", ch]";

            weightNames[position + 2] = "UGRNN Node [Layer " + layer + ", number " + number + ", gw]";
            weightNames[position + 3] = "UGRNN Node [Layer " + layer + ", number " + number + ", cw]";

            weightNames[position + 4] = "UGRNN Node [Layer " + layer + ", number " + number + ", gb]";
            weightNames[position + 5] = "UGRNN Node [Layer " + layer + ", number " + number + ", cb]";

            weightCount += 6;
        }

        for (Edge edge : outputEdges) {
            String targetType = "";
            if (edge.outputNode instanceof UGRNNNode) targetType = "UGRNN ";
            weightNames[position + weightCount] = "Edge from UGRNN Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + edge.outputNode.layer + ", number " + edge.outputNode.number + "]";
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            String targetType = "";
            if (recurrentEdge.outputNode instanceof UGRNNNode) targetType = "UGRNN ";

            weightNames[position + weightCount] = "Recurrent Edge from UGRNN Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + recurrentEdge.outputNode.layer + ", number " + recurrentEdge.outputNode.number + "]";
            weightCount++;
        }


        return weightCount;
    }



    /**
     * We need to override the getWeights from RecurrentNode as
     * an UGRNNNode will have 6 weights and biases as opposed to
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
            weights[position] = gh;
            weights[position + 1] = ch;
            weights[position + 2] = gw;
            weights[position + 3] = cw;

            weights[position + 4] = gb;
            weights[position + 5] = cb;

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
     * an UGRNNNode will have 6 weights and biases as opposed to
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
            deltas[position] = delta_gh;
            deltas[position + 1] = delta_ch;
            deltas[position + 2] = delta_gw;
            deltas[position + 3] = delta_cw;

            deltas[position + 4] = delta_gb;
            deltas[position + 5] = delta_cb;

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
     * an UGRNNNode will have 6 weights and biases as opposed to
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
            gh = weights[position];
            ch = weights[position + 1];
            gw = weights[position + 2];
            cw = weights[position + 3];

            gb = weights[position + 4];
            cb = weights[position + 5];

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
     * This propagates the postActivationValue at this UGRNN node
     * to all it's output nodes.
     */
    public void propagateForward(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 4
        //NOTE: recurrent edges need to be propagated forward from this timeStep to
        //their targetNode at timeStep + the recurrentEdge's timeSkip

        //calculate all the values. input = this.preActivationValue, output(ht) = postActivationValue
        if (timeStep == 0){
            this.g[timeStep] = sigmoid(gw*this.preActivationValue[timeStep] + gh * 0 + gb);
            this.C[timeStep] = Math.tanh(cw*this.preActivationValue[timeStep] + ch * 0 + cb);
            //ht
            this.postActivationValue[timeStep] = g[timeStep] * 0 + C[timeStep] * (1 - g[timeStep]);
        }else{
            this.g[timeStep] = sigmoid(gw*this.preActivationValue[timeStep] + gh * this.postActivationValue[timeStep -1] + gb);
            this.C[timeStep] = Math.tanh(cw*this.preActivationValue[timeStep] + ch * this.postActivationValue[timeStep -1] + cb);
            //ht
            this.postActivationValue[timeStep] = g[timeStep] * this.postActivationValue[timeStep -1] + C[timeStep] * (1 - g[timeStep]);
        }


        //propagate the edge forward
        for (Edge edge : outputEdges){
            edge.outputNode.preActivationValue[timeStep] += edge.weight * this.postActivationValue[timeStep];
            }
        //propagate the recurrent edge forward
        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            UGRNNNode node = (UGRNNNode)recurrentEdge.outputNode;
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
        e = (1 - g[timeStep]) * d_ht * (1 - C[timeStep] * C[timeStep]);
        i = (number_ht - C[timeStep]) * d_ht * ((1 - g[timeStep]) * g[timeStep]);

        //bias deltas
        delta_gb += i;
        delta_cb += e;

        //weight deltas
        delta_gh += i * number_ht;
        delta_ch += e * number_ht;
        delta_gw += i * this.preActivationValue[timeStep];
        delta_cw += e * this.preActivationValue[timeStep];


        //input and previous output deltas
        if (timeStep >0){
            delta_ht[timeStep -1] = (gh * i) + (g[timeStep] * d_ht) + (e * ch);
            }

        this.delta[timeStep] = (gw * i) + (cw * e);


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

        //set up the weight and bias in UGRNN node

        gh = rand.nextGaussian();
        ch = rand.nextGaussian();
        gw = rand.nextGaussian();
        cw = rand.nextGaussian();

        // bias values for the UGRNN node
        gb = rand.nextGaussian();
        cb = rand.nextGaussian() - 1;

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

        gh = rand.nextGaussian();
        ch = rand.nextGaussian();
        gw = rand.nextGaussian();
        cw = rand.nextGaussian();

        // bias values for the UGRNN node
        gb = rand.nextGaussian();
        cb = rand.nextGaussian() - 1;

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
        return "[UGRNN Node - layer: " + layer + ", number: " + number + ", type: " + nodeType + "]";
    }
}
