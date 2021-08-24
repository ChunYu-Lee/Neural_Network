/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.Arrays;
import java.util.Random;

import data.Sequence;
import data.TimeSeriesDataSet;
import data.TimeSeries;

import network.LossFunction;
import network.DeltaNode;
import network.NodeType;
import network.RecurrentNeuralNetwork;
import network.RNNNodeType;
import network.NeuralNetworkException;

import util.Log;



public class PA24TestsDelta {
    public static final int NUMBER_REPEATS = 1;
    public static final boolean generatingTestValues = false; //do not modify or it will overwrite my correctly generated test values

    public static void main(String[] arguments) {
        try {
            //test an Delta node with one time step to make sure
            //a basic forward and backward pass is working (without data
            //from previous or future time steps
            testDeltaForwardPass(3253 /*seed*/, 1 /*maxSequenceLength*/);

            //test an Delta node with 5 time steps to make sure that
            //the forward and backward pass is working for when there
            //are multiple time steps
            testDeltaForwardPass(9283 /*seed*/, 5 /*maxSequenceLength*/);

            //test an Delta node with 5 time steps to make sure that
            //the forward and backward pass is working for when there
            //are multiple time steps
            testDeltaForwardPass(12323 /*seed*/, 10 /*maxSequenceLength*/);
        } catch (NeuralNetworkException e) {
            System.err.println("Delta tests failed with exception: " + e);
            e.printStackTrace();
            System.exit(1);
        }

        //now that we've tested the Delta cell itself lets make sure it works
        //correctly inside of an RNN

        TimeSeriesDataSet dataSet = new TimeSeriesDataSet("flights data set",
                //new String[]{"./datasets/flight_0.csv", "./datasets/flight_1.csv", "./datasets/flight_2.csv", "./datasets/flight_3.csv"}, /* input file names */
                new String[]{"./datasets/flight_0_short.csv", "./datasets/flight_1_short.csv", "./datasets/flight_2_short.csv", "./datasets/flight_3_short.csv"}, /* inp    ut file names */
                new String[]{"AltAGL", "E1 RPM", "IAS", "LatAc", "NormAc", "Pitch", "Roll"}, /*parameter names for RNN input values */
                new String[]{"Pitch", "Roll"} /*parameter names for RNN target output values */
                );

        double[] mins = dataSet.getMins();
        double[] maxs = dataSet.getMaxs();

        Log.info("Data set had the following column mins: " + Arrays.toString(mins));
        Log.info("Data set had the following column maxs: " + Arrays.toString(maxs));

        //don't uncomment these as it will overwrite my precomputed correct values
        if (generatingTestValues) TestValues.writeArray(mins, "pa24_mins", 0, 0);
        if (generatingTestValues) TestValues.writeArray(maxs, "pa24_maxs", 0, 0);

        try {
            Log.info("Checking normalization column mins");
            TestValues.testArray(mins, TestValues.readArray("pa24_mins", 0, 0), "pa24_mins", 0, 0);
            Log.info("normalization mins were correct.");

            Log.info("Checking normalization column maxs");
            TestValues.testArray(maxs, TestValues.readArray("pa24_maxs", 0, 0), "pa24_maxs", 0, 0);
            Log.info("normalization maxs were correct.");

        } catch (NeuralNetworkException e) {
            Log.fatal("Normalization not correctly implemented, calcualted the wrong normalization min and max values: " + e);
            e.printStackTrace();
            System.exit(1);
        }



        dataSet.normalizeMinMax(mins, maxs);
        Log.info("normalized the data");


        // //test these with random initialization seeds and indexes
        // //do not change these numbers as I've saved expected results files
        // //for each of these
        // PA23Tests.testOneLayerBackwardPass("oneLayerDeltaTest", dataSet, RNNNodeType.Delta, "feed forward", LossFunction.NONE, 12345, 0);
        // PA23Tests.testOneLayerBackwardPass("oneLayerDeltaTest", dataSet, RNNNodeType.Delta, "feed forward", LossFunction.L2_NORM, 12345, 0);
        // PA23Tests.testOneLayerBackwardPass("oneLayerDeltaTest", dataSet, RNNNodeType.Delta, "jordan", LossFunction.L2_NORM, 13231, 2);
        // PA23Tests.testOneLayerBackwardPass("oneLayerDeltaTest", dataSet, RNNNodeType.Delta, "elman", LossFunction.L1_NORM, 19823, 1);

        // PA23Tests.testTwoLayerBackwardPass("twoLayerDeltaTest", dataSet, RNNNodeType.Delta, "feed forward", LossFunction.NONE, 18323, 0);
        // PA23Tests.testTwoLayerBackwardPass("twoLayerDeltaTest", dataSet, RNNNodeType.Delta, "feed forward", LossFunction.L1_NORM, 18323, 0);
        // PA23Tests.testTwoLayerBackwardPass("twoLayerDeltaTest", dataSet, RNNNodeType.Delta, "jordan", LossFunction.L1_NORM, 25142, 2);
        // PA23Tests.testTwoLayerBackwardPass("twoLayerDeltaTest", dataSet, RNNNodeType.Delta, "elman", LossFunction.L2_NORM, 2918382, 0);
    }

    public static void testDeltaForwardPass(int seed, int maxSequenceLength) throws NeuralNetworkException {
        int layer = 1;
        int number = 1;
        DeltaNode deltaNode = new DeltaNode(layer, number, NodeType.HIDDEN, maxSequenceLength);

        Random generator = new Random(seed);
        double[] weights = new double[6];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (generator.nextDouble() * 0.10) - 0.05;
        }

        deltaNode.setWeights(0, weights);

        //save the input values so we can use them to calculate the numeric gradient for the Delta node
        double[] inputValues = new double[maxSequenceLength];
        for (int i = 0; i < maxSequenceLength; i++) {
            inputValues[i] = generator.nextDouble();

            deltaNode.preActivationValue[i] = inputValues[i];
        }

        // for (int i = 0; i < maxSequenceLength; i++) {
        //     deltaNode.propagateForward(i);

        //     Log.debug("deltaNode time step " + i);
        //     Log.debug("\tdeltaNode.preActivationValue[" + i + "]: " + deltaNode.preActivationValue[i]);
        //     Log.debug("\tdeltaNode.postActivationValue[" + i + "]: " + deltaNode.postActivationValue[i]);
        //     Log.debug("\tdeltaNode.C[" + i + "]: " + deltaNode.C[i]);
        //     Log.debug("\tdeltaNode.rt[" + i + "]: " + deltaNode.rt[i]);
        //     Log.debug("\tdeltaNode.zt[" + i + "]: " + deltaNode.zt[i]);

        //     //do not uncomment these as they will overwrite the correct values I've generated for the test
        //     if (generatingTestValues) TestValues.writeValue(deltaNode.preActivationValue[i], "preActivationValue", seed, i, maxSequenceLength);
        //     if (generatingTestValues) TestValues.writeValue(deltaNode.postActivationValue[i], "postActivationValue", seed, i, maxSequenceLength);
        //     if (generatingTestValues) TestValues.writeValue(deltaNode.C[i], "C", seed, i, maxSequenceLength);
        //     if (generatingTestValues) TestValues.writeValue(deltaNode.rt[i], "rt", seed, i, maxSequenceLength);
        //     if (generatingTestValues) TestValues.writeValue(deltaNode.zt[i], "zt", seed, i, maxSequenceLength);

        //     Log.info("Checking preActivationValue for seed " + seed + ", sequenceIndex " + i + ", and maxSequenceLength: " + maxSequenceLength);
        //     TestValues.testValue(deltaNode.preActivationValue[i], TestValues.readValue("preActivationValue", seed, i, maxSequenceLength), "preActivationValue", seed, i, maxSequenceLength);

        //     Log.info("Checking postActivationValue for seed " + seed + ", sequenceIndex " + i + ", and maxSequenceLength: " + maxSequenceLength);
        //     TestValues.testValue(deltaNode.postActivationValue[i], TestValues.readValue("postActivationValue", seed, i, maxSequenceLength), "postActivationValue", seed, i, maxSequenceLength);

        //     Log.info("Checking C for seed " + seed + ", sequenceIndex " + i + ", and maxSequenceLength: " + maxSequenceLength);
        //     TestValues.testValue(deltaNode.C[i], TestValues.readValue("C", seed, i, maxSequenceLength), "C", seed, i, maxSequenceLength);

        //     Log.info("Checking rt for seed " + seed + ", sequenceIndex " + i + ", and maxSequenceLength: " + maxSequenceLength);
        //     TestValues.testValue(deltaNode.rt[i], TestValues.readValue("rt", seed, i, maxSequenceLength), "rt", seed, i, maxSequenceLength);

        //     Log.info("Checking zt for seed " + seed + ", sequenceIndex " + i + ", and maxSequenceLength: " + maxSequenceLength);
        //     TestValues.testValue(deltaNode.zt[i], TestValues.readValue("zt", seed, i, maxSequenceLength), "zt", seed, i, maxSequenceLength);

        // }


        double[] numericGradient = getDeltaNumericGradient(deltaNode, inputValues, weights, maxSequenceLength);
        Log.info("numeric gradient: " + Arrays.toString(numericGradient));

        //don't uncomment this as it will overwrite my precomputed correct values
        if (generatingTestValues) TestValues.writeArray(numericGradient, "numeric_gradient", seed, maxSequenceLength);

        // Log.info("Checking numeric_gradient for seed " + seed + ", and maxSequenceLength: " + maxSequenceLength);
        // TestValues.testArray(numericGradient, TestValues.readArray("numeric_gradient", seed, maxSequenceLength), "numeric_gradient", seed, maxSequenceLength);

        getDeltaOutput(deltaNode, inputValues, weights, maxSequenceLength);

        for (int i = 0; i < maxSequenceLength; i++ ) {
            //set the deltas to pseudo-random outputs so we can
            //test the backwards pass
            deltaNode.delta[i] = 1.0;
        }

        for (int i = maxSequenceLength - 1; i >= 0; i--) {
            deltaNode.propagateBackward(i);
        }

        double[] deltas = new double[6];
        deltaNode.getDeltas(0, deltas);
        Log.debug("delta_v: " + deltaNode.delta_v);
        Log.debug("delta_alpha: " + deltaNode.delta_alpha);
        Log.debug("delta_beta_1: " + deltaNode.delta_beta_1);
        Log.debug("delta_beta_2: " + deltaNode.delta_beta_2);

        Log.debug("delta_rb: " + deltaNode.delta_rb);
        Log.debug("delta_zhb: " + deltaNode.delta_zhb);


        for (int j = 0; j < deltas.length; j++) {
            Log.debug("deltaNode.deltas[" + j + "]: " + deltas[j]);
        }

        String[] weightNames = new String[6];
        deltaNode.getWeightNames(0, weightNames);


        Log.info("checking to see if numeric gradient and backprop deltas are close enough.");
        if (!BasicTests.gradientsCloseEnough(numericGradient, deltas, weightNames)) {
            throw new NeuralNetworkException("backprop vs numeric gradient check failed for seed " + seed + " and maxSequenceLength" + maxSequenceLength);
        }
    }

    public static double getDeltaOutput(DeltaNode deltaNode, double[] inputs, double[] weights, int maxSequenceLength) {
        deltaNode.reset();
        deltaNode.setWeights(0, weights);

        for (int i = 0; i < maxSequenceLength; i++) {
            deltaNode.preActivationValue[i] = inputs[i];
        }

        for (int i = 0; i < maxSequenceLength; i++) {
            deltaNode.propagateForward(i);
        }

        double outputSum = 0.0;
        for (int i = 0; i < maxSequenceLength; i++) {
            outputSum += deltaNode.postActivationValue[i];
        }

        return outputSum;
    }

    public static double[] getDeltaNumericGradient(DeltaNode deltaNode, double[] inputs, double[] weights, int maxSequenceLength) {
        double[] numericGradient = new double[weights.length];
        double[] testWeights = new double[weights.length];

        double H = 0.0000001;
        for (int i = 0; i < numericGradient.length; i++) {
            System.arraycopy(weights, 0, testWeights, 0, weights.length);

            testWeights[i] = weights[i] + H;
            double error1 = getDeltaOutput(deltaNode, inputs, testWeights, maxSequenceLength);

            testWeights[i] = weights[i] - H;
            double error2 = getDeltaOutput(deltaNode, inputs, testWeights, maxSequenceLength);

            numericGradient[i] = (error1 - error2) / (2.0 * H);

            Log.trace("numericGradient[" + i + "]: " + numericGradient[i] + ", error1: " + error1 + ", error2: " + error2 + ", testWeight1: " + (weights[i] + H) + ", testWeight2: "     + (weights[i] - H));
        }

        return numericGradient;
    }
}

