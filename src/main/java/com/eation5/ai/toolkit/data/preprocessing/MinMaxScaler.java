package com.eation5.ai.toolkit.data.preprocessing;

public class MinMaxScaler implements Preprocessor {
    private double minVal;
    private double maxVal;

    @Override
    public double[][] process(double[][] data) {
        if (data == null || data.length == 0) {
            return new double[0][0];
        }

        // Find min and max for each feature
        int numFeatures = data[0].length;
        double[] featureMin = new double[numFeatures];
        double[] featureMax = new double[numFeatures];

        for (int j = 0; j < numFeatures; j++) {
            featureMin[j] = data[0][j];
            featureMax[j] = data[0][j];
            for (int i = 1; i < data.length; i++) {
                if (data[i][j] < featureMin[j]) {
                    featureMin[j] = data[i][j];
                }
                if (data[i][j] > featureMax[j]) {
                    featureMax[j] = data[i][j];
                }
            }
        }

        // Apply min-max scaling
        double[][] scaledData = new double[data.length][numFeatures];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (featureMax[j] - featureMin[j] == 0) {
                    scaledData[i][j] = 0; // Avoid division by zero, all values are the same
                } else {
                    scaledData[i][j] = (data[i][j] - featureMin[j]) / (featureMax[j] - featureMin[j]);
                }
            }
        }
        return scaledData;
    }
}
