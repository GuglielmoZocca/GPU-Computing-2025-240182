#include "../include/time_lib.h"

//Arithmetic mean
double mu_fn_sol(double *v, int len) {

    double mu = 0.0;
    for (int i=0; i<len; i++)
        mu += (double)v[i];
    mu /= (double)len;

    return(mu);
}

// function to calculate geometric mean
double geometricMean(double arr[], int n)
{
    // declare product variable and
    // initialize it to 1.
    double product = 1;

    // Compute the product of all the
    // elements in the array.
    for (int i = 0; i < n; i++)
        product = product * arr[i];

    // compute geometric mean through formula
    // pow(product, 1/n) and return the value
    // to main function.
    double gm = pow(product, (double)1 / n);
    return gm;
}

//Arithmetic Standard deviation
double sigma_fn_sol(double *v, double mu, int len) {

    double sigma = 0.0;
    for (int i=0; i<len; i++) {
        sigma += ((double)v[i] - mu)*((double)v[i] - mu);
    }
    sigma /= (double)len;

    return(sigma);
}

//Geometric Standard deviation
double geometricStandardDeviation(const double* data,double geometricMean, int n) {
    if (n == 0 || geometricMean <= 0) return 0.0;

    double sum_sq = 0.0;

    for (size_t i = 0; i < n; ++i) {
        if (data[i] <= 0) {
            return -1;
        }
        double log_ratio = std::log(data[i] / geometricMean);
        sum_sq += log_ratio * log_ratio;
    }

    double variance = sum_sq / n;  // Use n-1 for sample GSD if needed
    return std::exp(std::sqrt(variance));
}

// -------------------------------------------------
