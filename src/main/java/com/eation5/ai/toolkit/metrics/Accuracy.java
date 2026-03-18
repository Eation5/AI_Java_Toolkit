package com.eation5.ai.toolkit.metrics;

public class Accuracy {
    public static double calculate(int[] actual, int[] predicted) {
        if (actual == null || predicted == null || actual.length != predicted.length) {
            throw new IllegalArgumentException("Actual and predicted arrays must be non-null and have the same length.");
        }

        int correct = 0;
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] == predicted[i]) {
                correct++;
            }
        }
        return (double) correct / actual.length;
    }
}
