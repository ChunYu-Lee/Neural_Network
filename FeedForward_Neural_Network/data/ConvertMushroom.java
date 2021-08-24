package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import util.Log;

public class ConvertMushroom {

    public static int searchIndex(int classNumber, char[] array, char c){
        for(int i=0; i < array.length; i++){
            if( array[i] == c){
                return i;
                }
            }

        System.err.println("ERROR! could not find class '" + c + "' in classes[" + classNumber + "]: " + Arrays.toString(array));
        System.exit(1);
        return -1;
        }

    public static void main(String[] arguments) {
        try {
            //create a buffered reader given the filename (which requires creating a File and FileReader object beforehand)
            BufferedReader bufferedReader = new BufferedReader(new FileReader(new File("./datasets/agaricus-lepiota.data")));
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File("./datasets/agaricus-lepiota_my_answer.txt")));
            
            //1 class and 22 attributes
            char[][] classes = new char[23][];
            //class
            classes[0] = new char[]{'e','p'};
            //attributes
            classes[1] = new char[]{'b', 'c', 'x', 'f', 'k', 's'};
            classes[2] = new char[]{'f', 'g', 'y', 's'};
            classes[3] = new char[]{'n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'};
            classes[4] = new char[]{'t', 'f'};
            classes[5] = new char[]{'a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'};
            classes[6] = new char[]{'a', 'd', 'f', 'n'};
            classes[7] = new char[]{'c', 'w', 'd'};
            classes[8] = new char[]{'b', 'n'};
            classes[9] = new char[]{'k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'};
            classes[10] = new char[]{'e', 't'};
            classes[11] = new char[]{'b', 'c','u', 'e', 'z', 'r', '?'};
            classes[12] = new char[]{'f', 'y', 'k', 's'};
            classes[13] = new char[]{'f', 'y', 'k', 's'};
            classes[14] = new char[]{'n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'};
            classes[15] = new char[]{'n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'};
            classes[16] = new char[]{'p', 'u'};
            classes[17] = new char[]{'n', 'o', 'w', 'y'};
            classes[18] = new char[]{'n', 'o', 't'};
            classes[19] = new char[]{'c', 'e', 'f', 'l', 'n', 'p', 's', 'z'};
            classes[20] = new char[]{'k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'};
            classes[21] = new char[]{'a', 'c', 'n', 's', 'v', 'y'};
            classes[22] = new char[]{'g', 'l', 'm', 'p', 'u', 'w', 'd'};
            
            String readLine = "";
            //read the file line by line
            while ((readLine = bufferedReader.readLine()) != null) {
               Log.info(readLine); //print out each line of the file if the -DLOG_LEVEL=DEBUG system property is set on the command line

                if (readLine.length() == 0 || readLine.charAt(0) == '#') {
                    //empty lines are skipped, as well as lines beginning
                    //with the '#' character, these are comments and also skipped
                    continue;
                }

                String[] values = readLine.split(",");

                //put everything in a stringbuffer before writing to the file
                StringBuffer sb = new StringBuffer();
                
                for (int i=0; i < classes.length; i++){
                    //class
                    if (i == 0){
                        if(values[i].charAt(0) == 'e'){
                            sb.append("0");
                        }else{
                            sb.append("1");
                            }
                        sb.append(":");
                    }else{ 
                        //attributes
                        for (int j=0 ; j < classes[i].length; j++){

                            if( j == searchIndex(i, classes[i], values[i].charAt(0)) ){
                                sb.append("1");
                            }else{
                                sb.append("0");
                                }
                            
                            if( i == (classes.length - 1) & j == (classes[i].length -1) ){
                                continue;
                            }else{
                                sb.append(",");
                                }
                            }
                        }
                    } 
                sb.append("\n");

                Log.info(sb.toString());
                bufferedWriter.write(sb.toString());
            }
            bufferedWriter.close();
            bufferedReader.close();

        } catch (IOException e) {
            Log.fatal("ERROR converting agaricus-lepiota data file");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
