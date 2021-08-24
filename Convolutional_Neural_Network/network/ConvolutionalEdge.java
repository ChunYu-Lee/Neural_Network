/**
 * This class represents an ConvolutionalEdge in a neural network. It will contain
 * the ConvolutionalEdge's weight, and also have references to input node and output
 * nodes of this edge.
 */
package network;

import java.util.Random;

import util.Log;

public class ConvolutionalEdge extends Edge {
    //the weight for this edge
    public double weight[][][];

    //the delta calculated by backpropagation for this edge
    public double weightDelta[][][];

    /**
     * This constructs a new edge in the neural network between the passed
     * parameters. It will register itself at the input and output nodes
     * through the Node.addOutgoingConvolutionalEdge(ConvolutionalEdge) and Node.addIncomingConvolutionalEdge(ConvolutionalEdge)
     * methods.
     *
     * @param inputNode is the input for this edge
     * @param outputNode is the output for this edge
     */
    public ConvolutionalEdge(ConvolutionalNode inputNode, ConvolutionalNode outputNode, int sizeZ, int sizeY, int sizeX) throws NeuralNetworkException {
        super(inputNode, outputNode, sizeZ, sizeY, sizeX);
        this.inputNode = inputNode;
        this.outputNode = outputNode;

        if (inputNode.sizeZ - sizeZ + 1 != outputNode.sizeZ
                || inputNode.sizeY - sizeY + 1 != outputNode.sizeY - (2 * outputNode.padding)
                || inputNode.sizeX - sizeX + 1 != outputNode.sizeX - (2 * outputNode.padding)) {
            throw new NeuralNetworkException("Cannot connect input node " + inputNode.toString() + " to output node " + outputNode.toString() + " because sizes do not work with this filter (" + sizeZ + "x" + sizeY + "x" + sizeX  + "), output node size should be (batchSize x" + (inputNode.sizeZ - sizeZ + 1) + "x" + (inputNode.sizeY - sizeY + 1) + "x" + (inputNode.sizeX - sizeX + 1) + ")");
        }

        //initialize the weight and delta to 0
        weight = new double[sizeZ][sizeY][sizeX];
        weightDelta = new double[sizeZ][sizeY][sizeX];
    }

    /**
     * Resets the deltas for this edge
     */
    public void reset() {
        //Log.info("resetting convolutional edge with sizeZ: " + sizeZ + ", sizeY: " + sizeY + ", sizeX: " + sizeX);
        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weightDelta[z][y][x] = 0;
                }
            }
        }
    }

    /**
     * Used to get the weights of this Edge.
     * It will set the weights in the weights
     * parameter passed in starting at position, and return the number of
     * weights it set.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weights is the array of weights we're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeights(int position, double[] weights) {
        int weightCount = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weights[position + weightCount] = weight[z][y][x];
                    weightCount++;
                }
            }
        }

        return weightCount;
    }

    /**
     * Used to print gradients related to this edge, along with informationa
     * about this edge.
     * It start printing the gradients passed in starting at position, and
     * return the number of gradients it printed.
     *
     * @param position is the index to start printing different gradients
     * @param numericGradient is the array of the numeric gradient we're printing
     * @param backpropGradient is the array of the backprop gradient we're printing
     *
     * @return the number of gradients printed by this edge
     */
    public int printGradients(int position, double[] numericGradient, double[] backpropGradient) {
        //don't print anything out, but print out this edge
        Log.info("ConvolutionalEdge from Node [layer: " + inputNode.layer + ", number: " + inputNode.number + "] to Node [layer: " + outputNode.layer + ", number: " + outputNode.number + "] to Node:");

        int count = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    Log.info("\tweights[" + z + "][" + y + "][" + x + "]: "+ Log.twoGradients(numericGradient[position + count], backpropGradient[position + count]));
                    count++;
                }
            }
        }

        return count;
    }


    /**
     * Used to get the deltas of this Edge.
     * It will set the deltas in the deltas
     * parameter passed in starting at position, and return the number of
     * deltas it set.
     *
     * @param position is the index to start setting deltas in the deltas parameter
     * @param deltas is the array of deltas we're setting.
     *
     * @return the number of deltas set in the deltas parameter
     */
    public int getDeltas(int position, double[] deltas) {
        int deltaCount = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    deltas[position + deltaCount] = weightDelta[z][y][x];
                    deltaCount++;
                }
            }
        }

        return deltaCount;
    }


    /**
     * Used to set the weights of this Edge.
     * It uses the same technique as Node.getWeights
     * where the starting position of weights to set is passed, and it returns
     * how many weights were set.
     *
     * @param position is the starting position in the weights parameter to start
     * setting weights from.
     * @param weights is the array of weights we are setting from
     *
     * @return the number of weights gotten from the weights parameter
     */
    public int setWeights(int position, double[] weights) {
        int weightCount = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weight[z][y][x] = weights[position + weightCount];
                    weightCount++;
                }
            }
        }

        return weightCount;
    }



    /**
     * This initializes the weights of this ConvolutionalEdge (Filter) by
     * the range calculated by it's output node (which should be sqrt(2)/sqrt(all incoming edge filter sizes).
     *
     * @param range is sqrt(2)/sqrt(sum of output node incoming filter sizes)
     */
    public void initializeKaiming(double range, int fanIn) {
        //TODO: Implement this for Programming Assignment 3 - Part 1

        //set up the weight in the edges
        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weight[z][y][x] = range;
                    }
                }
            }
    }

    /**
     * This initializes the weights of this ConvolutionalEdge (Filter) by
     * uniformly within the range calculated by it's output node (which
     * should be between negative and positive sqrt(6)/sqrt(all incoming
     * and outgoing edge filter sizes).
     *
     * @param range is sqrt(6)/sqrt(sum of output node incoming and outgoing filter sizes)
     */
    public void initializeXavier(double range, int fanIn, int fanOut) {
        //TODO: Implement this for Programming Assignment 3 - Part 1

        //set up the weight in the edges
        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weight[z][y][x] = range;
                    }
                }
            }
    }


    /**
     * This takes an incoming value from the input node
     * and propagates it forwards to the output node.
     *
     * @param inputValues is the value from the input node.
     */
    public void propagateForward(double[][][][] inputValues) {
        //TODO: You need to implement this for Programming Assignment 3 - Part 1
        int ystart = outputNode.padding, ystop = outputNode.sizeY - outputNode.padding;
        int xstart = outputNode.padding, xstop = outputNode.sizeX - outputNode.padding;

        for (int i = 0; i< outputNode.batchSize; i++){
            for (int z =0; z < outputNode.sizeZ; z++){
                for (int y = ystart; y < ystop; y++){
                    for (int x = xstart; x < xstop; x++){
                        double sum = 0.0;
                        int wz =0;

                        for (int zi = z; zi < sizeZ + z; zi++){
                            int wy =0;
                            int yiStart = y - outputNode.padding, yiStop = sizeY + y - outputNode.padding;

                            for (int yi = yiStart; yi < yiStop; yi++){
                                int wx=0;
                                int xiStart = x - outputNode.padding, xiStop = sizeX + x - outputNode.padding;
                                for (int xi = xiStart; xi <xiStop; xi++){
                                    sum += weight[wz][wy][wx] * inputValues[i][zi][yi][xi];
                                    wx++;
                                }
                                wy++;
                            }
                            wz++;
                        }
                        outputNode.inputValues[i][z][y][x] += sum;
                    }
                }
            }

        }
    }


    /**
     * This takes an incoming delta from the output node
     * and propagates it backwards to the input node.
     *
     * @param delta is the delta/error from the output node.
     */
    public void propagateBackward(double[][][][] delta) {
        //TODO: You need to implement this for Programming Assignment 3 - Part 2
        
        //outputNode delta location info
        for (int i = outputNode.batchSize -1; i>= 0; i--){
            for (int z= outputNode.sizeZ -1; z >= 0 ; z--){
                for (int y= outputNode.sizeY - outputNode.padding*2 -1; y >= 0; y--){
                    for (int x= outputNode.sizeX - outputNode.padding*2 -1; x >= 0;x--){
                        //filters
                        for (int zi = sizeZ + z-1; zi >= z; zi--){
                            for (int yi = sizeY + y-1; yi >= y; yi--){
                                for (int xi = sizeX + x-1; xi >= x; xi--){
                                    //here is the problem when input and output size change
                                    inputNode.delta[i][zi][yi][xi] += weight[zi -z][yi -y][xi-x] * delta[i][z][y][x];
                                    //System.out.println("inputnode :"+ i + ":" + z +":" + y +":"+ "x :"+  inputNode.delta[i][z][y][x]);
                                    weightDelta[zi - z][yi -y][xi -x] += inputNode.outputValues[i][zi][yi][xi] * delta[i][z][y][x];
                                    //System.out.println("weightDelta:" + weightDelta[zi-z][yi-y][xi-x]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }
}
