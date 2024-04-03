
// Idea behind radix sort is to iterate over each digit (left to right) and order
// Not an in place algorithm
__host__ radix_sort(unsigned int* input, unsigned int* output, unisgned int n, unsigned int sigFigs) {

    for (int i = 0; i < sigFigs; i++) {

        // 2 * num of blocks
        unsigned int onesPerBlock[2*(n / SECTION_SIZE)] = {0};
        unsigned int blockPositions[2*(n / SECTION_SIZE)] = {0};

        cudaMalloc((void**)&input, sizeof(input) / sizeof(input[0]));
        cudaMalloc((void**)&output, sizeof(output) / sizeof(output[0]));

        radix_sort_iter_get_bits<<<n / SECTION_SIZE, SECTION_SIZE>>>(input, bits, key, n, i);

        // exclusive scan on ones / zeros in each block. Could change to parallel but likely too few blocks for it to make a big difference
        blockPositions[0] = 0;
        for (unsigned int j = 0; j < n / SECTION_SIZE; j++) {
            for (unsigned int k = 0; k < blockDim.x; k++) if (bits[j][k] == 1) { onesPerBlock[j] += 1; } else { onesPerBlock[j + (n / SECTION_SIZE)] += 1; }
        }

        for (unsigned int j = 1; j < n / SECTION_SIZE; j++) {
            blockPositions[i] = blockPositions[i - 1] + onesPerBlock[i - 1];
        }

        radix_sort_iter_assign<<<n / SECTION_SIZE, SECTION_SIZE>>>(bits, output, blockPositions, n);

        // Repeat with output the new input
        input = output;

        cudaFree(input);
        cudaFree(output);
    }

}

// Sorts input within each block
__global__ void radix_sort_iter_get_bits(unsigned int* input unsigned int* bits, unisgned int n, unsigned int iter) {
    unsigned int i = threadIdx.x;
    __shared__ unsigned int bits[blockDim.x];
    unsigned int bit, key;
    if (i < n) {
        key = input[i];
        bit = (key >> iter) & 1;
        bits[i] = bit;
    }

    __syncthreads();


    // Sort internally
    __shared__ unsigned int exclsuiveScannedBits[blockDim.x];

    if (i < n) {
        for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
            __syncthreads();
            float temp;
            if (threadIdx.x >= stride) {
                temp = bits[i] + bits[i - stride];
            }
            __syncthreads();
            if (threadIdx.x >= stride) {
                bits[i] = temp;
            }
        }
        if (i < n) {
            exclsuiveScannedBits[i] = bits[i];
        }
    }

    __syncthreads();

    if (i < n) {
        unsigned int numOnesBefore = (i > 0) ? exclsuiveScannedBits[i - 1] : 0;
        unsigned int numOnesTotal = exclsuiveScannedBits[n - 1];
        unsigned int dst = (bit == 0) ? (blockIdx*blockDim + threadIdx - numOnesBefore) : ((blockIdx + 1)*blockDim + threadIdx - numOnesTotal - numOnesBefore);
        
        input[dst] = key;
    }
}

// Global sort
__global__ void radix_sort_iter_assign(unsigned int* input, unsigned int* output, unsigned int* blockPositions, unisgned int n) {
    unsigned int i = threadIdx.x;
    if (i < n) {
        unsigned int dst = (bit == 0) ? (blockPositions[blockIdx.x] + i) : (blockPositions[blockIdx.x + (n / SECTION_SIZE)] + i);
        
        output[dst] = input[i];
    }
}