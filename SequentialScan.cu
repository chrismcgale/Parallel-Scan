__host__ __global__ void sequential_inclusive_scan(float *x, float *y, unsigned int n) {
    y[0] = x[0];
    for (unsigned int i = 1; i < n; i++) {
        y[i] = y[i - 1] + x[i];
    }
}

__host__ __global__ void sequential_exclusive_scan(float *x, float *y, unsigned int n) {
    y[0] = 0;
    for (unsigned int i = 1; i < n; i++) {
        y[i] = y[i - 1] + x[i - 1];
    }
}