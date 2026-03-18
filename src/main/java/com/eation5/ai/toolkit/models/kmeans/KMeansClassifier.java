package com.eation5.ai.toolkit.models.kmeans;

import com.eation5.ai.toolkit.data.Dataset;
import com.eation5.ai.toolkit.models.Model;

import java.util.Arrays;
import java.util.Random;

public class KMeansClassifier implements Model {
    private int k;
    private int maxIterations;
    private double[][] centroids;
    private Random random;

    public KMeansClassifier(int k, int maxIterations) {
        this.k = k;
        this.maxIterations = maxIterations;
        this.random = new Random();
    }

    @Override
    public void train(Dataset dataset) {
        double[][] features = dataset.getFeatures();
        if (features == null || features.length == 0) {
            return;
        }

        int numSamples = features.length;
        int numFeatures = features[0].length;

        // Initialize centroids randomly
        centroids = new double[k][numFeatures];
        for (int i = 0; i < k; i++) {
            int randomIndex = random.nextInt(numSamples);
            centroids[i] = Arrays.copyOf(features[randomIndex], numFeatures);
        }

        int[] assignments = new int[numSamples];
        boolean changed = true;
        int iteration = 0;

        while (changed && iteration < maxIterations) {
            changed = false;
            // Assign each sample to the closest centroid
            for (int i = 0; i < numSamples; i++) {
                int closestCentroid = getClosestCentroid(features[i]);
                if (closestCentroid != assignments[i]) {
                    assignments[i] = closestCentroid;
                    changed = true;
                }
            }

            // Update centroids
            double[][] newCentroids = new double[k][numFeatures];
            int[] clusterCounts = new int[k];

            for (int i = 0; i < numSamples; i++) {
                int cluster = assignments[i];
                for (int j = 0; j < numFeatures; j++) {
                    newCentroids[cluster][j] += features[i][j];
                }
                clusterCounts[cluster]++;
            }

            for (int i = 0; i < k; i++) {
                if (clusterCounts[i] > 0) {
                    for (int j = 0; j < numFeatures; j++) {
                        centroids[i][j] = newCentroids[i][j] / clusterCounts[i];
                    }
                } else {
                    // If a cluster becomes empty, reinitialize its centroid
                    int randomIndex = random.nextInt(numSamples);
                    centroids[i] = Arrays.copyOf(features[randomIndex], numFeatures);
                    changed = true; // Force another iteration
                }
            }
            iteration++;
        }
    }

    @Override
    public int[] predict(double[][] features) {
        if (centroids == null || centroids.length == 0) {
            throw new IllegalStateException("Model has not been trained.");
        }
        int[] predictions = new int[features.length];
        for (int i = 0; i < features.length; i++) {
            predictions[i] = getClosestCentroid(features[i]);
        }
        return predictions;
    }

    private int getClosestCentroid(double[] sample) {
        double minDistance = Double.MAX_VALUE;
        int closestCentroid = -1;

        for (int i = 0; i < k; i++) {
            double distance = euclideanDistance(sample, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroid = i;
            }
        }
        return closestCentroid;
    }

    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
}
