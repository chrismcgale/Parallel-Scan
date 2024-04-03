__host__  hierchical_scan(float *X, float *Y, unsigned int n, bool inclusive = true) {
    float S[n/SECTION_SIZE];
    cudaMalloc((void**)&X, sizeof(float));
    cudaMalloc((void**)&Y, sizeof(float));
    cudaMalloc((void**)&S, sizeof(float));

    hierchical_scan_kernel<<<n / SECTION_SIZE, SECTION_SIZE>>>(X, Y, S, n, inclusive);

    Kogge_Stone_scan_kernel<<<1, n / SECTION_SIZE>>>(S, S, n / SECTION_SIZE, true);

    hierchical_scan_final_kernel<<<n / SECTION_SIZE, SECTION_SIZE>>>(S, Y);

    cudaFree(X);
    cudaFree(Y);
    cudaFree(S);

    return 0;
}

// Same as KoggeStone but also writes partial sums to S. Could use BrentKung instead.
__global__ hierchical_scan_kernel(float *X, float *Y, float *S, unsigned int n, bool inclusive = true) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n || (!inclusive && threadIdx.x = 0)) {
        if (inclusive) {
            XY[threadIdx.x] = X[i];
        } else {
            XY[threadIdx.x] = X[i - 1];
        }
    } else {
        XY[threadIdx.x] = 0.0f;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if (threadIdx.x >= stride) {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            XY[threadIdx.x] = temp;
        }
    }
    if (i < n) {
        Y[i] = XY[threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x == blockDim.x - 1) {
        S[blockIdx.x] = XY[SECTION_SIZE - 1]
    }
}

__global__ hierchical_scan_final_kernel(float *S, float *Y) {
    unsigned int i = blockIdx.x*blockDim + threadIdx.x;
    Y[i] += S[blockIdx.x - 1];
}