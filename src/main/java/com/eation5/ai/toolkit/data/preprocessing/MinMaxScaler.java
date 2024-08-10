package com.eation5.ai.toolkit.data.preprocessing;

import java.util.Arrays;

public class MinMaxScaler implements Preprocessor {
    private double[] featureMin;
    private double[] featureMax;
    private double scaleMin;
    private double scaleMax;

    /**
     * Constructor for MinMaxScaler with default scaling range [0, 1].
     */
    public MinMaxScaler() {
        this(0.0, 1.0);
    }

    /**
     * Constructor for MinMaxScaler with a custom scaling range.
     * @param scaleMin The minimum value of the output range.
     * @param scaleMax The maximum value of the output range.
     */
    public MinMaxScaler(double scaleMin, double scaleMax) {
        if (scaleMin >= scaleMax) {
            throw new IllegalArgumentException("scaleMin must be less than scaleMax.");
        }
        this.scaleMin = scaleMin;
        this.scaleMax = scaleMax;
    }

    /**
     * Fits the scaler to the input data by calculating the min and max for each feature.
     * @param data The input data (samples x features).
     */
    public void fit(double[][] data) {
        if (data == null || data.length == 0 || data[0].length == 0) {
            throw new IllegalArgumentException("Input data cannot be empty.");
        }

        int numFeatures = data[0].length;
        featureMin = new double[numFeatures];
        featureMax = new double[numFeatures];

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
    }

    /**
     * Transforms the input data using the fitted min and max values.
     * @param data The input data to transform.
     * @return The scaled data.
     * @throws IllegalStateException if the scaler has not been fitted yet.
     */
    public double[][] transform(double[][] data) {
        if (featureMin == null || featureMax == null) {
            throw new IllegalStateException("Scaler has not been fitted. Call fit() first.");
        }
        if (data == null || data.length == 0 || data[0].length == 0) {
            return new double[0][0];
        }
        if (data[0].length != featureMin.length) {
            throw new IllegalArgumentException("Number of features in data does not match the fitted scaler.");
        }

        double[][] scaledData = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                double range = featureMax[j] - featureMin[j];
                if (range == 0) {
                    scaledData[i][j] = scaleMin; // All values are the same, map to scaleMin
                } else {
                    scaledData[i][j] = ((data[i][j] - featureMin[j]) / range) * (scaleMax - scaleMin) + scaleMin;
                }
            }
        }
        return scaledData;
    }

    /**
     * Fits the scaler to the data and then transforms it.
     * This method is part of the Preprocessor interface.
     * @param data The input data to fit and transform.
     * @return The scaled data.
     */
    @Override
    public double[][] process(double[][] data) {
        fit(data);
        return transform(data);
    }

    // Optional: Inverse transform method
    public double[][] inverseTransform(double[][] scaledData) {
        if (featureMin == null || featureMax == null) {
            throw new IllegalStateException("Scaler has not been fitted. Call fit() first.");
        }
        if (scaledData == null || scaledData.length == 0 || scaledData[0].length == 0) {
            return new double[0][0];
        }
        if (scaledData[0].length != featureMin.length) {
            throw new IllegalArgumentException("Number of features in scaled data does not match the fitted scaler.");
        }

        double[][] originalData = new double[scaledData.length][scaledData[0].length];
        for (int i = 0; i < scaledData.length; i++) {
            for (int j = 0; j < scaledData[0].length; j++) {
                double range = featureMax[j] - featureMin[j];
                if (range == 0) {
                    originalData[i][j] = featureMin[j]; // All values were the same
                } else {
                    originalData[i][j] = ((scaledData[i][j] - scaleMin) / (scaleMax - scaleMin)) * range + featureMin[j];
                }
            }
        }
        return originalData;
    }

    @Override
    public String toString() {
        return "MinMaxScaler{" +
               "featureMin=" + Arrays.toString(featureMin) +
               ", featureMax=" + Arrays.toString(featureMax) +
               ", scaleMin=" + scaleMin +
               ", scaleMax=" + scaleMax +
               '}';
    }
}
