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

public class DeltaNode extends RecurrentNode {

    //these are the weight values(4) for the Delta node
    double v;
    double alpha;
    double beta_1;
    double beta_2;

    //these are the bias values(2) for the Delta node
    double rb;
    double zhb;

    //these are the deltas for the weights and biases
    public double delta_v;
    public double delta_alpha;
    public double delta_beta_1;
    public double delta_beta_2;

    public double delta_rb;
    public double delta_zhb;
    
    //zt is the output --> postActivationValue
    //this is the delta value for zt in the diagram, it will be
    //set to the sum of the delta coming in from the outputs (delta)
    //plus whatever deltas came in from the subsequent time step during backprop
    public double[] delta_zt;

    //rt values for each time step
    public double[] rt;

    //variable d1 saved for doing the backward pass
    public double[] d1;

    //variable zh1 saved for doing the backward pass
    public double[] zh1;

    //variable zh2 saved for doing the backward pass
    public double[] zh2;

    //variable zh3 saved for doing the backward pass
    public double[] zh3;

    //variable zc saved for doing the backward pass
    public double[] zc;

    //variable z1 saved for doing the backward pass
    public double[] z1;

    //variable z2 saved for doing the backward pass
    public double[] z2;



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
    public DeltaNode(int layer, int number, NodeType nodeType, int maxSequenceLength) {
        super(layer, number, nodeType, maxSequenceLength, null);

        delta_zt = new double[maxSequenceLength];

        rt = new double[maxSequenceLength];
        d1 = new double[maxSequenceLength];
        zh1 = new double[maxSequenceLength];
        zh2 = new double[maxSequenceLength];
        zh3 = new double[maxSequenceLength];
        zc = new double[maxSequenceLength];
        z1 = new double[maxSequenceLength];
        z2 = new double[maxSequenceLength];

   }

    /**
     * This resets the values which need to be recalcualted for
     * each forward and backward pass. It will also reset the
     * deltas for outgoing nodes.
     */
    public void reset() {
        //use RecurrentNode's reset to reset everything this has inherited from
        //RecurrentNode, then reset the DeltaNode's fields
        super.reset();
        Log.trace("Resetting Delta node: " + toString());

        for (int timeStep = 0; timeStep < maxSequenceLength; timeStep++) {
            rt[timeStep] = 0;
            d1[timeStep] = 0;
            zh1[timeStep] = 0;
            zh2[timeStep] = 0;
            zh3[timeStep] = 0;
            zc[timeStep] = 0;
            z1[timeStep] = 0;
            z2[timeStep] = 0;

            delta_zt[timeStep] = 0;
        }

        delta_v = 0;
        delta_alpha = 0;
        delta_beta_1 = 0;
        delta_beta_2 = 0;

        delta_rb = 0;
        delta_zhb = 0;

    }


    /**
     * We need to override the getWeightNames from RecurrentNode as
     * an DeltaNode will have 6 weight and bias names as opposed to
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
            weightNames[position] = "Delta Node [layer " + layer + ", number " + number + ", v]";
            weightNames[position + 1] = "Delta Node [Layer " + layer + ", number " + number + ", alpha]";
            weightNames[position + 2] = "Delta Node [Layer " + layer + ", number " + number + ", beta_1]";
            weightNames[position + 3] = "Delta Node [Layer " + layer + ", number " + number + ", beta_2]";

            weightNames[position + 4] = "Delta Node [Layer " + layer + ", number " + number + ", rb]";
            weightNames[position + 5] = "Delta Node [Layer " + layer + ", number " + number + ", zhb]";

            weightCount += 6;
        }

        for (Edge edge : outputEdges) {
            String targetType = "";
            if (edge.outputNode instanceof DeltaNode) targetType = "Delta ";
            weightNames[position + weightCount] = "Edge from Delta Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + edge.outputNode.layer + ", number " + edge.outputNode.number + "]";
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            String targetType = "";
            if (recurrentEdge.outputNode instanceof DeltaNode) targetType = "Delta ";

            weightNames[position + weightCount] = "Recurrent Edge from Delta Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + recurrentEdge.outputNode.layer + ", number " + recurrentEdge.outputNode.number + "]";
            weightCount++;
        }


        return weightCount;
    }



    /**
     * We need to override the getWeights from RecurrentNode as
     * an DeltaNode will have 6 weights and biases as opposed to
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
            weights[position] = v;
            weights[position + 1] = alpha;
            weights[position + 2] = beta_1;
            weights[position + 3] = beta_2;

            weights[position + 4] = rb;
            weights[position + 5] = zhb;

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
     * an DeltaNode will have 6 weights and biases as opposed to
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
            deltas[position] = delta_v;
            deltas[position + 1] = delta_alpha;
            deltas[position + 2] = delta_beta_1;
            deltas[position + 3] = delta_beta_2;

            deltas[position + 4] = delta_rb;
            deltas[position + 5] = delta_zhb;

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
     * an DeltaNode will have 6 weights and biases as opposed to
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
            v = weights[position];
            alpha = weights[position + 1];
            beta_1 = weights[position + 2];
            beta_2 = weights[position + 3];

            rb = weights[position + 4];
            zhb = weights[position + 5];

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
     * This propagates the postActivationValue at this Delta node
     * to all it's output nodes.
     */
    public void propagateForward(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 4
        //NOTE: recurrent edges need to be propagated forward from this timeStep to
        //their targetNode at timeStep + the recurrentEdge's timeSkip

        //calculate all the values. input = this.preActivationValue, output(ht) = postActivationValue
        if (timeStep == 0){
            this.rt[timeStep] = sigmoid(this.preActivationValue[timeStep] + rb);
            this.d1[timeStep] = v * 0;
            this.zh1[timeStep] = alpha * d1[timeStep] * this.preActivationValue[timeStep];
            this.zh2[timeStep] = beta_1 * d1[timeStep];
            this.zh3[timeStep] = beta_2 * this.preActivationValue[timeStep];
            this.zc[timeStep] = Math.tanh(zh1[timeStep] + zh2[timeStep] + zh3[timeStep]+zhb);
            this.z1[timeStep] = (1 - rt[timeStep]) * zc[timeStep];
            this.z2[timeStep] = rt[timeStep] * 0; 
            
            //zt
            this.postActivationValue[timeStep] = Math.tanh(z1[timeStep] + z2[timeStep]);

        }else{
            this.rt[timeStep] = sigmoid(this.preActivationValue[timeStep] + rb);
            this.d1[timeStep] = v * this.postActivationValue[timeStep -1];
            this.zh1[timeStep] = alpha * d1[timeStep] * this.preActivationValue[timeStep];
            this.zh2[timeStep] = beta_1 * d1[timeStep];
            this.zh3[timeStep] = beta_2 * this.preActivationValue[timeStep];
            this.zc[timeStep] = Math.tanh(zh1[timeStep] + zh2[timeStep] + zh3[timeStep]+zhb);
            this.z1[timeStep] = (1 - rt[timeStep]) * zc[timeStep];
            this.z2[timeStep] = rt[timeStep] * this.postActivationValue[timeStep -1]; 
            
            //zt
            this.postActivationValue[timeStep] = Math.tanh(z1[timeStep] + z2[timeStep]);
            
            }


        //propagate the edge forward
        for (Edge edge : outputEdges){
            edge.outputNode.preActivationValue[timeStep] += edge.weight * this.postActivationValue[timeStep];
            }
        //propagate the recurrent edge forward
        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            DeltaNode node = (DeltaNode)recurrentEdge.outputNode;
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
        double t = 0.0;
        double h = 0.0;
        double i = 0.0;
        
        //delta zt
        double d_zt = 0.0;

        //zt-1
        double number_zt = 0.0;

        //last timeStep
        if (timeStep == maxSequenceLength - 1){
            d_zt = 0 + this.delta[timeStep];
        }else{
            d_zt = delta_zt[timeStep] + this.delta[timeStep];
        }

        if (timeStep > 0){
            number_zt = this.postActivationValue[timeStep - 1];
        }

        //variable update
        t = d_zt * (1 - (this.postActivationValue[timeStep] * this.postActivationValue[timeStep]));
        h = t * (1 - rt[timeStep]) *(1 - (zc[timeStep] * zc[timeStep]));
        i = t * (number_zt - zc[timeStep]) * rt[timeStep] * (1 - rt[timeStep]);

        //bias deltas
        delta_rb += i;
        delta_zhb += h;

        //weight deltas
        delta_v += h * (beta_1 + alpha * this.preActivationValue[timeStep]) * number_zt;
        delta_alpha += h * this.preActivationValue[timeStep] * d1[timeStep];
        delta_beta_1 += h * d1[timeStep];
        delta_beta_2 += h * this.preActivationValue[timeStep];

        //input and previous output deltas
        if (timeStep >0){
            delta_zt[timeStep -1] = (t * rt[timeStep]) + (h * (beta_1 + this.preActivationValue[timeStep] * alpha) * v);
            }

        this.delta[timeStep] = i + h * (beta_2 + alpha * d1[timeStep]);


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

        //set up the weight and bias in Delta node

        v = rand.nextGaussian();
        alpha = rand.nextGaussian();
        beta_1 = rand.nextGaussian();
        beta_2 = rand.nextGaussian();

        // bias values for the Delta node
        rb = rand.nextGaussian()-1;
        zhb = rand.nextGaussian();

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
        

        v = rand.nextGaussian();
        alpha = rand.nextGaussian();
        beta_1 = rand.nextGaussian();
        beta_2 = rand.nextGaussian();

        // bias values for the Delta node
        rb = rand.nextGaussian()-1;
        zhb = rand.nextGaussian();

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
        return "[Delta Node - layer: " + layer + ", number: " + number + ", type: " + nodeType + "]";
    }
}
