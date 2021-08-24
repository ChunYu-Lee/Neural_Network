/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

import data.DataSet;
import data.Instance;

import network.LossFunction;
import network.NeuralNetwork;
import network.NeuralNetworkException;

import util.Log;
import util.Vector;


public class PA14GradientDescent {
    public static void helpMessage() {
        Log.info("Usage:");
        Log.info("\tjava PA14GradientDescent <data set> <gradient descent type> <batch size> <loss function> <epochs> <bias> <learning rate> <mu> <adaptive learning method> <eps> <layer_size_1 ... layer_size_n");
        Log.info("\t\tdata set can be: 'and', 'or' or 'xor', 'iris' or 'mushroom'");
        Log.info("\t\tgradient descent type can be: 'stochastic', 'minibatch' or 'batch'");
        Log.info("\t\tbatch size should be > 0. Will be ignored for stochastic or batch gradient descent");
        Log.info("\t\tloss function can be: 'l1_norm', 'l2_norm', 'svm' or 'softmax'");
        Log.info("\t\tepochs is an integer > 0");
        Log.info("\t\tbias is a double");
        Log.info("\t\tlearning rate is a double usually small and > 0");
        Log.info("\t\tmu is a double < 1 and typical values are 0.5, 0.9, 0.95 and 0.99");
        Log.info("\t\tadaptive learning method can be: 'nesterov', 'adagrad', 'rmsprop', 'adam'");
        Log.info("\t\teps is typically around 1e-4 to 1e-8, Will be ignored for nesterov method");
        Log.info("\t\tdacayRate is between 0 and 1. Will be ignored for nesterov, adagrad, and adam methods");
        Log.info("\t\tbeta1 is recommended 0.9. It is only for adam method");
        Log.info("\t\tbeta2 is recommended 0.999. It is only for adam method");
        Log.info("\t\tlayer_size_1..n is a list of integers which are the number of nodes in each hidden layer");

    }

    public static void main(String[] arguments) {
        if (arguments.length < 14) {
            helpMessage();
            System.exit(1);
        }

        String dataSetName = arguments[0];
        String descentType = arguments[1];
        int batchSize = Integer.parseInt(arguments[2]);
        String lossFunctionName = arguments[3];
        int epochs = Integer.parseInt(arguments[4]);
        double bias = Double.parseDouble(arguments[5]);
        double learningRate = Double.parseDouble(arguments[6]);
        double mu = Double.parseDouble(arguments[7]);
        String adaptiveLearningName = arguments[8];
        double eps = Double.parseDouble(arguments[9]);
        double decayRate = Double.parseDouble(arguments[10]);
        double beta1 = Double.parseDouble(arguments[11]);
        double beta2 = Double.parseDouble(arguments[12]);

        //after add the adaptive learning rate, eps, decayRate, beta1, and beta1, we have to move from 8 to 14.
        int[] layerSizes = new int[arguments.length - 13]; // the remaining arguments are the layer sizes
        for (int i = 13; i < arguments.length; i++) {
            layerSizes[i - 13] = Integer.parseInt(arguments[i]);
        }

        //the and, or and xor datasets will have 1 output (the number of output columns)
        //but the iris and  mushroom datasets will have the number of output classes
        int outputLayerSize = 0;

        DataSet dataSet = null;
        if (dataSetName.equals("and")) {
            dataSet = new DataSet("and data", "./datasets/and.txt");
            outputLayerSize = dataSet.getNumberOutputs();
        } else if (dataSetName.equals("or")) {
            dataSet = new DataSet("or data", "./datasets/or.txt");
            outputLayerSize = dataSet.getNumberOutputs();
        } else if (dataSetName.equals("xor")) {
            dataSet = new DataSet("xor data", "./datasets/xor.txt");
            outputLayerSize = dataSet.getNumberOutputs();
        } else if (dataSetName.equals("iris")) {
            //TODO: PA1-4: Make sure you implement the getInputMeans,
            //getInputStandardDeviations and normalize methods in
            //DataSet to get this to work.
            dataSet = new DataSet("iris data", "./datasets/iris.txt");
            double[] means = dataSet.getInputMeans();
            double[] stdDevs = dataSet.getInputStandardDeviations();
            Log.info("data set means: " + Arrays.toString(means));
            Log.info("data set standard deviations: " + Arrays.toString(stdDevs));
            dataSet.normalize(means, stdDevs);

            outputLayerSize = dataSet.getNumberClasses();
        } else if (dataSetName.equals("mushroom")) {
            dataSet = new DataSet("mushroom data", "./datasets/agaricus-lepiota.txt");
            outputLayerSize = dataSet.getNumberClasses();
        } else {
            Log.fatal("unknown data set : " + dataSetName);
            System.exit(1);
        }

        LossFunction lossFunction = LossFunction.NONE;
        if (lossFunctionName.equals("l1_norm")) {
            Log.info("Using an L1_NORM loss function.");
            lossFunction = LossFunction.L1_NORM;
        } else if (lossFunctionName.equals("l2_norm")) {
            Log.info("Using an L2_NORM loss function.");
            lossFunction = LossFunction.L2_NORM;
        } else if (lossFunctionName.equals("svm")) {
            Log.info("Using an SVM loss function.");
            lossFunction = LossFunction.SVM;
        } else if (lossFunctionName.equals("softmax")) {
            Log.info("Using an SOFTMAX loss function.");
            lossFunction = LossFunction.SOFTMAX;
        } else {
            Log.fatal("unknown loss function : " + lossFunctionName);
            System.exit(1);
        }

        NeuralNetwork nn = new NeuralNetwork(dataSet.getNumberInputs(), layerSizes, outputLayerSize, lossFunction);
        try {
            nn.connectFully();
        } catch (NeuralNetworkException e) {
            Log.fatal("ERROR connecting the neural network -- this should not happen!.");
            e.printStackTrace();
            System.exit(1);
        }

        //start the gradient descent
        try {
            Log.info("Starting " + descentType + " gradient descent!");
            if (descentType.equals("minibatch")) {
                Log.info(descentType + "(" + batchSize + "), " + dataSetName + ", " + lossFunctionName + ", lr: " + learningRate + ", mu:" + mu);
            } else {
                Log.info(descentType + ", " + dataSetName + ", " + lossFunctionName + ", lr: " + learningRate + ", mu:" + mu);
            }

            nn.initializeRandomly(bias);

            //TODO: For PA1-4 use this and implement nesterov momentum
            //java will initialize each element in the array to 0
            double[] velocity = new double[nn.getNumberWeights()];

            //TODO: BONUS PA1-4: (1 point) implement the RMSprop
            //per-parameter adaptive learning rate method.
            //TODO: BONUS PA1-4: (1 point) implement the Adam
            //per-parameter adaptive learning rate method.
            //For these you will need to add a command line flag
            //to select if which method you'll use (nesterov, rmsprop or adam)

            double bestError = 10000;
            double error = nn.forwardPass(dataSet.getInstances()) / dataSet.getNumberInstances();
            double accuracy = nn.calculateAccuracy(dataSet.getInstances());

            if (error < bestError) bestError = error;
            System.out.println("  " + bestError + " " + error + " " + String.format("%10.5f", accuracy * 100.0) /*make hte accuracy a percentage*/);

            for (int i = 0; i < epochs; i++) {

                if (descentType.equals("stochastic")) {
                    //TODO: PA1-3 you need to implement one epoch (pass through the
                    //training data) for stochastic gradient descent

                    //shuffle the instances
                    dataSet.shuffle();

                    //get the gradient and update the weights
                    for(int k=0; k < dataSet.getNumberInstances(); k++){

                        double gradient[] = nn.getGradient(dataSet.getInstance(k));
                        double temp_weights[] = nn.getWeights();
                        for(int j=0; j < temp_weights.length; j++){

                            if (adaptiveLearningName.equals("nesterov")){

                                //current velocity
                                double velocityPrev = velocity[j];
                                //next velocity
                                velocity[j] = mu * velocity[j] - learningRate * gradient[j];
                                //update current weight
                                temp_weights[j] += ((-mu * velocityPrev) + (1 + mu) * velocity[j]);

                            }else if(adaptiveLearningName.equals("adagrad")){

                                //setup the cache[]
                                double[] cache = new double[nn.getNumberWeights()];
                                //run the adagrad
                                cache[j] += (gradient[j] * gradient[j]);
                                temp_weights[j] -= (learningRate / (Math.sqrt(cache[j] + eps)) * gradient[j]);

                            }else if(adaptiveLearningName.equals("rmsprop")){

                                //setup the cache[]
                                double[] cache = new double[nn.getNumberWeights()];
                                //run the rmsprop
                                cache[j] = decayRate * cache[j] + (1 - decayRate) * (gradient[j] * gradient[j]);
                                temp_weights[j] -= (learningRate / Math.sqrt(cache[j] + eps)) * gradient[j];

                            }else if(adaptiveLearningName.equals("adam")){

                                //implement Adam with bias correction
                                //set up m[], mt[], v[], vt[]
                                double[] m = new double[nn.getNumberWeights()];
                                //double[] mt = new double[nn.getNumberWeights()];
                                double[] v = new double[nn.getNumberWeights()];
                                //double[] vt = new double[nn.getNumberWeights()];

                                m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];
                                // 1- beta1^(epoch)
                                //mt[j] = m[j] / (1 - Math.pow(beta1, i));
                                v[j] = beta2 * v[j] + (1 - beta2) * (gradient[j] * gradient[j]);
                                //vt[j] = v[j] / (1 - Math.pow(beta2, i));

                                //weight update
                                temp_weights[j] -= (learningRate * m[j] / Math.sqrt(v[j] +eps));

                            }else{
                                Log.fatal("unknown adaptive learning: " + adaptiveLearningName);
                                System.exit(1);
                                }
                            }
                    //set the updated weight in the nn
                        nn.setWeights(temp_weights);
                        }

                } else if (descentType.equals("minibatch")) {
                    //TODO: PA1-3 you need to implement one epoch (pass through the
                    //training data) for minibatch gradient descent

                    //shuffle the instances
                    dataSet.shuffle();
                    int NumberInstances = dataSet.getNumberInstances();
                    double[] gradient = new double [nn.getNumberWeights()];

                    for(int l=0; l < NumberInstances ; l += batchSize){

                        double temp_weights[] = nn.getWeights();
                        
                        //operates till the last bacth
                        if (l < NumberInstances - batchSize){

                            List<Instance> instances = dataSet.getInstances(l, batchSize);
                            gradient = nn.getGradient(instances);

                        }else{
                        //check whether the # of instances % batchSize ==0, if not then edge case
                            if ( NumberInstances % batchSize ==0){

                                List<Instance> instances = dataSet.getInstances(l, batchSize);
                                gradient = nn.getGradient(instances);

                            
                            }else{
                                
                                int edgeNumber = NumberInstances - l;
                                List<Instance> instances = dataSet.getInstances(l, edgeNumber);
                                gradient = nn.getGradient(instances);
                                
                                }
                        }

                        for(int j=0; j < temp_weights.length; j++){
                            //start the learning rate method
                            if (adaptiveLearningName.equals("nesterov")){

                                //current velocity
                                double velocityPrev = velocity[j];
                                //next velocity
                                velocity[j] = mu * velocity[j] - learningRate * gradient[j];
                                //update current weight
                                temp_weights[j] += ((-mu * velocityPrev) + (1 + mu) * velocity[j]);

                            }else if(adaptiveLearningName.equals("adagrad")){

                                //setup the cache[]
                                double[] cache = new double[nn.getNumberWeights()];
                                //run the adagrad
                                cache[j] += (gradient[j] * gradient[j]);
                                temp_weights[j] -= (learningRate / (Math.sqrt(cache[j] + eps)) * gradient[j]);

                            }else if(adaptiveLearningName.equals("rmsprop")){

                                //setup the cache[]
                                double[] cache = new double[nn.getNumberWeights()];
                                //run the rmsprop
                                cache[j] = decayRate * cache[j] + (1 - decayRate) * (gradient[j] * gradient[j]);
                                temp_weights[j] -= (learningRate / Math.sqrt(cache[j] + eps)) * gradient[j];

                            }else if(adaptiveLearningName.equals("adam")){

                                //implement Adam with bias correction
                                //set up m[], mt[], v[], vt[]
                                double[] m = new double[nn.getNumberWeights()];
                                //double[] mt = new double[nn.getNumberWeights()];
                                double[] v = new double[nn.getNumberWeights()];
                                //double[] vt = new double[nn.getNumberWeights()];

                                m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];
                                // 1- beta1^(epoch)
                                //mt[j] = m[j] / (1 - Math.pow(beta1, i));
                                v[j] = beta2 * v[j] + (1 - beta2) * (gradient[j] * gradient[j]);
                                //vt[j] = v[j] / (1 - Math.pow(beta2, i));

                                //weight update
                                temp_weights[j] -= (learningRate * m[j] / Math.sqrt(v[j] +eps));

                            }else{
                                Log.fatal("unknown adaptive learning: " + adaptiveLearningName);
                                System.exit(1);
                                }
                        }
                        nn.setWeights(temp_weights);
                    }
                } else if (descentType.equals("batch")) {
                    //TODO: PA1-3 you need to implement one epoch (pass through the training
                    //instances) for batch gradient descent

                    //get all the gradient and set up temp_weight to update the weight
                    double gradient[] = nn.getGradient(dataSet.getInstances());
                    double temp_weights[] = nn.getWeights();
                    
                    for(int j=0; j < temp_weights.length; j++){
                        
                        //start the learning rate method
                        if (adaptiveLearningName.equals("nesterov")){

                            //current velocity
                            double velocityPrev = velocity[j];
                            //next velocity
                            velocity[j] = mu * velocity[j] - learningRate * gradient[j];
                            //update current weight
                            temp_weights[j] += ((-mu * velocityPrev) + (1 + mu) * velocity[j]);

                        }else if(adaptiveLearningName.equals("adagrad")){

                            //setup the cache[]
                            double[] cache = new double[nn.getNumberWeights()];
                            //run the adagrad
                            cache[j] += (gradient[j] * gradient[j]);
                            temp_weights[j] -= (learningRate / (Math.sqrt(cache[j] + eps)) * gradient[j]);

                        }else if(adaptiveLearningName.equals("rmsprop")){

                            //setup the cache[]
                            double[] cache = new double[nn.getNumberWeights()];
                            //run the rmsprop
                            cache[j] = decayRate * cache[j] + (1 - decayRate) * (gradient[j] * gradient[j]);
                            temp_weights[j] -= (learningRate / Math.sqrt(cache[j] + eps)) * gradient[j];

                        }else if(adaptiveLearningName.equals("adam")){

                            //implement Adam with bias correction
                            //set up m[], mt[], v[], vt[]
                            double[] m = new double[nn.getNumberWeights()];
                            //double[] mt = new double[nn.getNumberWeights()];
                            double[] v = new double[nn.getNumberWeights()];
                            //double[] vt = new double[nn.getNumberWeights()];

                            m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];
                            // 1- beta1^(epoch)
                            //mt[j] = m[j] / (1 - Math.pow(beta1, i));
                            v[j] = beta2 * v[j] + (1 - beta2) * (gradient[j] * gradient[j]);
                            //vt[j] = v[j] / (1 - Math.pow(beta2, i));

                            //weight update
                            temp_weights[j] -= (learningRate * m[j] / Math.sqrt(v[j] +eps));

                        }else{
                            Log.fatal("unknown adaptive learning: " + adaptiveLearningName);
                            System.exit(1);
                            }



                        //set the updated weight in the nn
                        nn.setWeights(temp_weights);
                    }
                } else {
                    Log.fatal("unknown descent type: " + descentType);
                    helpMessage();
                    System.exit(1);
                }


                Log.info("weights: " + Arrays.toString(nn.getWeights()));

                //at the end of each epoch, calculate the error over the entire
                //set of instances and print it out so we can see if we're decreasing
                //the overall error
                error = nn.forwardPass(dataSet.getInstances()) / dataSet.getNumberInstances();
                accuracy = nn.calculateAccuracy(dataSet.getInstances());

                if (error < bestError) bestError = error;
                System.out.println(i + " " + bestError + " " + error + " " + String.format("%10.5f", accuracy * 100.0) /*make hte accuracy a percentage*/);
            }

        } catch (NeuralNetworkException e) {
            Log.fatal("gradient descent failed with exception: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }
}
