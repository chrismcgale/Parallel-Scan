#DEFINE BITS_PER_ITER 2;


// Idea behind radix sort is to iterate over each digit (left to right) and order
// Not an in place algorithm
__host__ radix_sort(unsigned int* input, unsigned int* output, unisgned int n, unsigned int sigFigs) {

    int rows = (int)pow(2, BITS_PER_ITER);

    for (int i = 0; i < sigFigs / BITS_PER_ITER; i++) {

        // 2 * num of blocks
        unsigned int onesPerBlock[rows*(n / SECTION_SIZE)] = {0};
        unsigned int blockPositions[rows*(n / SECTION_SIZE)] = {0};

        cudaMalloc((void**)&input, sizeof(input) / sizeof(input[0]));
        cudaMalloc((void**)&output, sizeof(output) / sizeof(output[0]));

        radix_sort_iter_get_bits<<<n / SECTION_SIZE, SECTION_SIZE>>>(input, bits, key, n, i);

        // exclusive scan on ones / zeros in each block. Could change to parallel but likely too few blocks for it to make a big difference
        blockPositions[0] = 0;
        for (unsigned int j = 0; j < n / SECTION_SIZE; j++) {
            for (unsigned int k = 0; k < blockDim.x; k++) onesPerBlock[j + ((n / SECTION_SIZE)) * bits[j][k] ] += 1;
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
        bit_set = (key >> iter) & ((1 << BITS_PER_ITER) - 1);
        bits[i] = bit_set;
    }

    __syncthreads();

    __shared__ unsigned int exclsuiveScannedBits[blockDim.x];
    if (i < n) {
        exclsuiveScannedBits[i] = bits[i];
    }
    __syncthreads();

    // One exclusive scan per bit is required
    if (i < n) {
        for (unsigned int j = 0; j < BITS_PER_ITER; j++) {
            for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
                __syncthreads();
                float temp;
                if (i >= stride) {
                    temp = exclsuiveScannedBits[i] + exclsuiveScannedBits[i - stride];
                }
                __syncthreads();
                if (i >= stride) {
                    exclsuiveScannedBits[i] = temp;
                }
            }

            __syncthreads();

            unsigned int numOnesBefore = (i > 0) ? exclsuiveScannedBits[i - 1] : 0;
            unsigned int numOnesTotal = exclsuiveScannedBits[n - 1];

            unsigned int dst = (bit == 0) ? (i - numOnesBefore) : (n - numOnesTotal - numOnesBefore);

            unsigned int temp = exclsuiveScannedBits[i];

            __syncthreads();

            exclsuiveScannedBits[dst] = temp;

            __syncthreads();
        

        }
        unsigned int dst = blockIdx*blockDim + pos;
        
        input[dst] = key;
    }

}



// Global sort
__global__ void radix_sort_iter_assign(unsigned int* input, unsigned int* output, unsigned int* blockPositions, unisgned int n) {
    unsigned int i = threadIdx.x;
    if (i < n) {
        unsigned int dst = blockPositions[blockIdx.x + ((n / SECTION_SIZE)) * bits[j][k]] + i;
        
        output[dst] = input[i];
    }
}