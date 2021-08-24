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

public class GRUNode extends RecurrentNode {

    //these are the weight values(6) for the GRU node
    double zw;
    double zu;

    double rw;
    double ru;

    double hw;
    double hu;

    //these are the bias values(3) for the GRU node
    double zb;
    double rb;
    double hb;

    //these are the deltas for the weights and biases
    public double delta_zw;
    public double delta_zu;

    public double delta_rw;
    public double delta_ru;

    public double delta_hw;
    public double delta_hu;

    public double delta_zb;
    public double delta_rb;
    public double delta_hb;

    //this is the delta value for ht in the diagram, it will be
    //set to the sum of the delta coming in from the outputs (delta)
    //plus whatever deltas came in from
    //the subsequent time step during backprop
    public double[] delta_ht;

    //input gate values for each time step
    public double[] zt;

    //forget gate values for each time step
    public double[] rt;

    //variable C saved for doing the backward pass
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
    public GRUNode(int layer, int number, NodeType nodeType, int maxSequenceLength) {
        super(layer, number, nodeType, maxSequenceLength, null);

        delta_ht = new double[maxSequenceLength];

        zt = new double[maxSequenceLength];
        rt = new double[maxSequenceLength];
        C = new double[maxSequenceLength];


   }

    /**
     * This resets the values which need to be recalcualted for
     * each forward and backward pass. It will also reset the
     * deltas for outgoing nodes.
     */
    public void reset() {
        //use RecurrentNode's reset to reset everything this has inherited from
        //RecurrentNode, then reset the GRUNode's fields
        super.reset();
        Log.trace("Resetting GRU node: " + toString());

        for (int timeStep = 0; timeStep < maxSequenceLength; timeStep++) {
            zt[timeStep] = 0;
            rt[timeStep] = 0;
            C[timeStep] = 0;
            delta_ht[timeStep] = 0;
        }

        delta_zw = 0;
        delta_zu = 0;

        delta_rw = 0;
        delta_ru = 0;

        delta_hw = 0;
        delta_hu = 0;

        delta_zb = 0;
        delta_rb = 0;
        delta_hb = 0;
    }


    /**
     * We need to override the getWeightNames from RecurrentNode as
     * an GRUNode will have 9 weight and bias names as opposed to
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
            weightNames[position] = "GRU Node [layer " + layer + ", number " + number + ", zw]";
            weightNames[position + 1] = "GRU Node [Layer " + layer + ", number " + number + ", zu]";

            weightNames[position + 2] = "GRU Node [Layer " + layer + ", number " + number + ", rw]";
            weightNames[position + 3] = "GRU Node [Layer " + layer + ", number " + number + ", ru]";

            weightNames[position + 4] = "GRU Node [Layer " + layer + ", number " + number + ", hw]";
            weightNames[position + 5] = "GRU Node [Layer " + layer + ", number " + number + ", hu]";

            weightNames[position + 6] = "GRU Node [Layer " + layer + ", number " + number + ", zb]";
            weightNames[position + 7] = "GRU Node [Layer " + layer + ", number " + number + ", rb]";
            weightNames[position + 8] = "GRU Node [Layer " + layer + ", number " + number + ", hb]";

            weightCount += 9;
        }

        for (Edge edge : outputEdges) {
            String targetType = "";
            if (edge.outputNode instanceof GRUNode) targetType = "GRU ";
            weightNames[position + weightCount] = "Edge from GRU Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + edge.outputNode.layer + ", number " + edge.outputNode.number + "]";
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            String targetType = "";
            if (recurrentEdge.outputNode instanceof GRUNode) targetType = "GRU ";

            weightNames[position + weightCount] = "Recurrent Edge from GRU Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + recurrentEdge.outputNode.layer + ", number " + recurrentEdge.outputNode.number + "]";
            weightCount++;
        }


        return weightCount;
    }



    /**
     * We need to override the getWeights from RecurrentNode as
     * an GRUNode will have 9 weights and biases as opposed to
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
            weights[position] = zw;
            weights[position + 1] = zu;

            weights[position + 2] = rw;
            weights[position + 3] = ru;

            weights[position + 4] = hw;
            weights[position + 5] = hu;

            weights[position + 6] = zb;
            weights[position + 7] = rb;
            weights[position + 8] = hb;

            weightCount += 9;
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
     * an GRUNode will have 9 weights and biases as opposed to
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
            deltas[position] = delta_zw;
            deltas[position + 1] = delta_zu;

            deltas[position + 2] = delta_rw;
            deltas[position + 3] = delta_ru;

            deltas[position + 4] = delta_hw;
            deltas[position + 5] = delta_hu;

            deltas[position + 6] = delta_zb;
            deltas[position + 7] = delta_rb;
            deltas[position + 8] = delta_hb;

            deltaCount += 9;
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
     * an GRUNode will have 9 weights and biases as opposed to
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
            zw = weights[position];
            zu = weights[position + 1];

            rw = weights[position + 2];
            ru = weights[position + 3];

            hw = weights[position + 4];
            hu = weights[position + 5];

            zb = weights[position + 6];
            rb = weights[position + 7];
            hb = weights[position + 8];

            weightCount += 9;
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
     * This propagates the postActivationValue at this GRU node
     * to all it's output nodes.
     */
    public void propagateForward(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 4
        //NOTE: recurrent edges need to be propagated forward from this timeStep to
        //their targetNode at timeStep + the recurrentEdge's timeSkip

        //calculate all the values. input = this.preActivationValue, output(ht) = postActivationValue
        if (timeStep == 0){
            this.zt[timeStep] = sigmoid(zw*this.preActivationValue[timeStep] + zu*0 + zb);
            this.rt[timeStep] = sigmoid(rw*this.preActivationValue[timeStep] + ru*0 + rb);
            this.C[timeStep] = Math.tanh(hw*this.preActivationValue[timeStep] + rt[timeStep]*hu*0 + hb);
            //ht
            this.postActivationValue[timeStep] = zt[timeStep]*0 + (1 - zt[timeStep])* C[timeStep];
        }else{
            this.zt[timeStep] = sigmoid(zw*this.preActivationValue[timeStep] + zu*this.postActivationValue[timeStep -1] + zb);
            this.rt[timeStep] = sigmoid(rw*this.preActivationValue[timeStep] + ru*this.postActivationValue[timeStep -1] + rb);
            this.C[timeStep] = Math.tanh(hw*this.preActivationValue[timeStep] + rt[timeStep]*hu*this.postActivationValue[timeStep -1] + hb);
            this.postActivationValue[timeStep] = zt[timeStep]*this.postActivationValue[timeStep -1] + (1 - zt[timeStep])* C[timeStep];
        }


        //propagate the edge forward
        for (Edge edge : outputEdges){
            edge.outputNode.preActivationValue[timeStep] += edge.weight * this.postActivationValue[timeStep];
            }
        //propagate the recurrent edge forward
        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            GRUNode node = (GRUNode)recurrentEdge.outputNode;
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
        double d = 0.0;
        double e = 0.0;
        double f = 0.0;
        double g = 0.0;
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
        d = d_ht * (1 - zt[timeStep]);
        e = d * (1 - (C[timeStep] * C[timeStep]));
        f = e * (rt[timeStep] * hu * number_ht) * (1 - rt[timeStep]);
        g = d_ht * (number_ht - C[timeStep]) * (zt[timeStep]*(1 - zt[timeStep]));

        //bias deltas
        delta_zb += g;
        delta_rb += f;
        delta_hb += e;

        //weight deltas
        delta_zw += g * this.preActivationValue[timeStep];
        delta_zu += g * number_ht;

        delta_rw += f * this.preActivationValue[timeStep];
        delta_ru += f * number_ht;

        delta_hw += e * this.preActivationValue[timeStep];
        delta_hu += e * rt[timeStep] * number_ht;

        //input and previous output deltas
        if (timeStep >0){
            delta_ht[timeStep -1] = (g * zu) + (f * ru) + (e * rt[timeStep] * hu) + (d_ht * zt[timeStep]);
            }

        this.delta[timeStep] = (e * hw) + (f * rw) + (g * zw);


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

        //set up the weight and bias in GRU node

        zw = rand.nextGaussian();
        zu = rand.nextGaussian();

        rw = rand.nextGaussian();
        ru = rand.nextGaussian();

        hw = rand.nextGaussian();
        hu = rand.nextGaussian();

        // bias values for the GRU node
        zb = rand.nextGaussian();
        rb = rand.nextGaussian()-1;
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

        zw = rand.nextGaussian();
        zu = rand.nextGaussian();

        rw = rand.nextGaussian();
        ru = rand.nextGaussian();

        hw = rand.nextGaussian();
        hu = rand.nextGaussian();

        // bias values for the GRU node
        zb = rand.nextGaussian();
        rb = rand.nextGaussian()-1;
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
        return "[GRU Node - layer: " + layer + ", number: " + number + ", type: " + nodeType + "]";
    }
}
