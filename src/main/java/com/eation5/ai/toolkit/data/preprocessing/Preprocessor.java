package com.eation5.ai.toolkit.data.preprocessing;

/**
 * Interface for data preprocessing techniques.
 * Implementations should provide methods to fit the preprocessor to data
 * and transform data based on the fitted parameters.
 */
public interface Preprocessor {

    /**
     * Fits the preprocessor to the input data. This method should calculate
     * any necessary parameters (e.g., min/max for scaling, mean/std for standardization).
     * @param data The input data (samples x features) to fit the preprocessor on.
     */
    void fit(double[][] data);

    /**
     * Transforms the input data using the parameters learned during the fit phase.
     * @param data The input data (samples x features) to transform.
     * @return The transformed data.
     */
    double[][] transform(double[][] data);

    /**
     * A convenience method that first fits the preprocessor to the data and then transforms it.
     * @param data The input data to fit and transform.
     * @return The transformed data.
     */
    default double[][] process(double[][] data) {
        fit(data);
        return transform(data);
    }
}
