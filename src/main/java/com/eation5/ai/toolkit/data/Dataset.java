package com.eation5.ai.toolkit.data;

import com.eation5.ai.toolkit.data.preprocessing.Preprocessor;

public class Dataset {
    private double[][] features;
    private int[] labels;

    public Dataset(double[][] features, int[] labels) {
        this.features = features;
        this.labels = labels;
    }

    public double[][] getFeatures() {
        return features;
    }

    public int[] getLabels() {
        return labels;
    }

    public void setFeatures(double[][] features) {
        this.features = features;
    }

    public void setLabels(int[] labels) {
        this.labels = labels;
    }

    public void applyPreprocessor(Preprocessor preprocessor) {
        this.features = preprocessor.process(this.features);
    }
}
