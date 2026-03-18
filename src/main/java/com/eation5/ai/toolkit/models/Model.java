package com.eation5.ai.toolkit.models;

import com.eation5.ai.toolkit.data.Dataset;

public interface Model {
    void train(Dataset dataset);
    int[] predict(double[][] features);
}
