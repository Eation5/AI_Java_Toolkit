package com.eation5.ai.toolkit.models.kmeans;

import com.eation5.ai.toolkit.data.Dataset;
import com.eation5.ai.toolkit.data.preprocessing.MinMaxScaler;
import com.eation5.ai.toolkit.metrics.Accuracy;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class KMeansClassifierTest {

    @Test
    void testKMeansClassification() {
        // Generate synthetic data for two clusters
        double[][] features = {
            {1.0, 1.0}, {1.2, 1.5}, {0.8, 1.1}, {1.5, 0.9}, // Cluster 0
            {5.0, 5.0}, {5.5, 5.2}, {4.8, 4.9}, {5.1, 5.3}  // Cluster 1
        };
        int[] labels = {0, 0, 0, 0, 1, 1, 1, 1}; // True labels

        Dataset dataset = new Dataset(features, labels);

        // Preprocess data (optional, but good practice)
        MinMaxScaler scaler = new MinMaxScaler();
        dataset.applyPreprocessor(scaler);

        // Initialize and train K-Means model
        KMeansClassifier model = new KMeansClassifier(2, 100); // 2 clusters, max 100 iterations
        model.train(dataset);

        // Make predictions on the training data
        int[] predictions = model.predict(dataset.getFeatures());

        // Evaluate the model - K-Means is unsupervised, so we check if clusters align with true labels
        // This test assumes that the two clusters found by K-Means will correspond to the two true labels.
        // Due to the nature of K-Means (random initialization), the cluster assignments (0 or 1) might be swapped.
        // We need to check for both possibilities.

        boolean case1 = true; // Predictions match labels directly
        boolean case2 = true; // Predictions are inverted (0->1, 1->0)

        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] != labels[i]) {
                case1 = false;
            }
            if (predictions[i] == labels[i]) {
                case2 = false;
            }
        }
        
        // If case1 is false, try to invert predictions for case2
        if (!case1) {
            case2 = true; // Reset case2 check
            for (int i = 0; i < predictions.length; i++) {
                // Invert prediction: if original was 0, inverted is 1; if original was 1, inverted is 0
                int invertedPrediction = (predictions[i] == 0) ? 1 : 0;
                if (invertedPrediction != labels[i]) {
                    case2 = false;
                    break;
                }
            }
        }

        assertTrue(case1 || case2, "K-Means clustering should correctly separate the two groups.");

        // Test with a new sample
        double[] newSampleCluster0 = {1.1, 1.3};
        double[] newSampleCluster1 = {5.3, 5.1};

        // Scale new samples using the same scaler
        double[][] scaledNewSample0 = scaler.process(new double[][]{newSampleCluster0});
        double[][] scaledNewSample1 = scaler.process(new double[][]{newSampleCluster1});

        int prediction0 = model.predict(scaledNewSample0)[0];
        int prediction1 = model.predict(scaledNewSample1)[0];

        // Verify that new samples are assigned to one of the two clusters
        assertTrue(prediction0 == 0 || prediction0 == 1, "New sample should be assigned to a cluster.");
        assertTrue(prediction1 == 0 || prediction1 == 1, "New sample should be assigned to a cluster.");

        // Further check if they are assigned to different clusters (assuming they should be)
        assertNotEquals(prediction0, prediction1, "Samples from different groups should be in different clusters.");
    }

    @Test
    void testEmptyDataset() {
        Dataset emptyDataset = new Dataset(new double[0][0], new int[0]);
        KMeansClassifier model = new KMeansClassifier(2, 10);
        model.train(emptyDataset);
        // No exception should be thrown, and centroids should remain null or empty
        assertNull(model.centroids, "Centroids should be null for an empty dataset.");
    }

    @Test
    void testSingleFeatureDataset() {
        double[][] features = {{1.0}, {1.5}, {5.0}, {8.0}};
        int[] labels = {0, 0, 1, 1};
        Dataset dataset = new Dataset(features, labels);
        MinMaxScaler scaler = new MinMaxScaler();
        dataset.applyPreprocessor(scaler);

        KMeansClassifier model = new KMeansClassifier(2, 100);
        model.train(dataset);
        int[] predictions = model.predict(dataset.getFeatures());

        boolean case1 = true;
        boolean case2 = true;

        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] != labels[i]) {
                case1 = false;
            }
            if (predictions[i] == labels[i]) {
                case2 = false;
            }
        }
        
        if (!case1) {
            case2 = true;
            for (int i = 0; i < predictions.length; i++) {
                int invertedPrediction = (predictions[i] == 0) ? 1 : 0;
                if (invertedPrediction != labels[i]) {
                    case2 = false;
                    break;
                }
            }
        }
        assertTrue(case1 || case2, "K-Means clustering should work with single-feature data.");
    }
}
