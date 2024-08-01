package com.eation5.ai.toolkit.data;

import com.eation5.ai.toolkit.data.preprocessing.Preprocessor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

public class Dataset implements Iterable<double[]> {
    private double[][] features;
    private int[] labels;
    private int currentBatchIndex = 0;

    public Dataset(double[][] features, int[] labels) {
        if (features == null || labels == null) {
            throw new IllegalArgumentException("Features and labels cannot be null.");
        }
        if (features.length != labels.length) {
            throw new IllegalArgumentException("Number of feature samples must match number of labels.");
        }
        this.features = features;
        this.labels = labels;
    }

    // Factory method to load data from a CSV file
    public static Dataset fromCsv(String filePath, int labelColumnIndex, boolean hasHeader) throws IOException {
        List<List<Double>> featureList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            if (hasHeader) {
                br.readLine(); // Skip header row
            }
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                List<Double> currentFeatures = new ArrayList<>();
                for (int i = 0; i < values.length; i++) {
                    if (i == labelColumnIndex) {
                        labelList.add(Integer.parseInt(values[i].trim()));
                    } else {
                        currentFeatures.add(Double.parseDouble(values[i].trim()));
                    }
                }
                featureList.add(currentFeatures);
            }
        }

        double[][] featuresArray = new double[featureList.size()][featureList.get(0).size()];
        for (int i = 0; i < featureList.size(); i++) {
            for (int j = 0; j < featureList.get(i).size(); j++) {
                featuresArray[i][j] = featureList.get(i).get(j);
            }
        }

        int[] labelsArray = labelList.stream().mapToInt(Integer::intValue).toArray();

        return new Dataset(featuresArray, labelsArray);
    }

    public double[][] getFeatures() {
        return features;
    }

    public int[] getLabels() {
        return labels;
    }

    public int size() {
        return features.length;
    }

    public int numFeatures() {
        return features.length > 0 ? features[0].length : 0;
    }

    public void applyPreprocessor(Preprocessor preprocessor) {
        this.features = preprocessor.process(this.features);
    }

    // Splits the dataset into training and testing sets
    public Dataset[] split(double testRatio) {
        if (testRatio < 0 || testRatio >= 1) {
            throw new IllegalArgumentException("Test ratio must be between 0 and 1 (exclusive).");
        }

        int totalSize = size();
        int testSize = (int) (totalSize * testRatio);
        int trainSize = totalSize - testSize;

        double[][] trainFeatures = new double[trainSize][];
        int[] trainLabels = new int[trainSize];
        double[][] testFeatures = new double[testSize][];
        int[] testLabels = new int[testSize];

        // Simple sequential split for now, can be randomized later
        for (int i = 0; i < trainSize; i++) {
            trainFeatures[i] = features[i];
            trainLabels[i] = labels[i];
        }
        for (int i = 0; i < testSize; i++) {
            testFeatures[i] = features[trainSize + i];
            testLabels[i] = labels[trainSize + i];
        }

        return new Dataset[]{new Dataset(trainFeatures, trainLabels), new Dataset(testFeatures, testLabels)};
    }

    // Provides an iterator for batch processing
    @Override
    public Iterator<double[]> iterator() {
        return new Iterator<double[]>() {
            private int currentIndex = 0;

            @Override
            public boolean hasNext() {
                return currentIndex < features.length;
            }

            @Override
            public double[] next() {
                if (!hasNext()) {
                    throw new java.util.NoSuchElementException();
                }
                return features[currentIndex++];
            }
        };
    }

    // Method to get a batch of data
    public Dataset getBatch(int batchSize) {
        if (currentBatchIndex >= features.length) {
            currentBatchIndex = 0; // Reset for next epoch
            return null; // No more batches
        }

        int endIndex = Math.min(currentBatchIndex + batchSize, features.length);
        int actualBatchSize = endIndex - currentBatchIndex;

        double[][] batchFeatures = new double[actualBatchSize][];
        int[] batchLabels = new int[actualBatchSize];

        for (int i = 0; i < actualBatchSize; i++) {
            batchFeatures[i] = features[currentBatchIndex + i];
            batchLabels[i] = labels[currentBatchIndex + i];
        }

        currentBatchIndex = endIndex;
        return new Dataset(batchFeatures, batchLabels);
    }

    // Reset batch index for new epoch
    public void resetBatchIndex() {
        this.currentBatchIndex = 0;
    }

    @Override
    public String toString() {
        return "Dataset{" +
               "numSamples=" + size() +
               ", numFeatures=" + numFeatures() +
               ", featuresShape=" + (features.length > 0 ? features.length + "x" + features[0].length : "0x0") +
               '}';
    }
}
