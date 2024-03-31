__global__ void Brent_Kung_scan_kernel(float *X, float *Y, unsigned int n, bool inclusive = true) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (inclusive) {
            XY[threadIdx.x] = X[i];
        } else {
            XY[threadIdx.x] = i > 0 ? X[i - 1] : 0.0f;
        }
    }
    if (i + blockDim.x < n) {
        if (inclusive) {
            XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
        } else {
            XY[threadIdx.x + blockDim.x] = i > 0 ? X[i + blockDim.x - 1] : 0.0f;
        } 
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1)*2*stride - 1;
        if (index < SECTION_SIZE) {
            XY[index] += XY[index - stride];
        }
    }
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1)*2*stride - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    if (i < n) Y[i] = XY[threadIdx.x];
    if (i + blockDim.x < n) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}