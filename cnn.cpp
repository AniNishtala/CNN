#include "header.h"
#include "weights.h"

// Convert every 3x3 patch into one column.
void Im2col(input_t input[INPUT_SIZE][INPUT_SIZE],
            input_t columns[FILTER_VALUES][OUTPUT_VALUES]) {
    #pragma HLS INLINE off

    for (int position = 0; position < OUTPUT_VALUES; position++) {
        #pragma HLS PIPELINE II=1

        int row = position / OUTPUT_SIZE;
        int col = position % OUTPUT_SIZE;

        for (int i = 0; i < KERNEL_SIZE; i++) {
            #pragma HLS UNROLL
            for (int j = 0; j < KERNEL_SIZE; j++) {
                #pragma HLS UNROLL

                int index = i * KERNEL_SIZE + j;
                columns[index][position] = input[row + i][col + j];
            }
        }
    }
}

// Perform convolution using the Im2col matrix and fixed weights.
void GemmConv2d0(input_t input[INPUT_SIZE][INPUT_SIZE],
                activation_t output[OUTPUT_SIZE][OUTPUT_SIZE]) {
    // Allow the pixels + weights to be read in parallel.
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=3 dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=3 dim=2
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=0

    input_t columns[FILTER_VALUES][OUTPUT_VALUES];
    #pragma HLS ARRAY_PARTITION variable=columns complete dim=1
    #pragma HLS STREAM variable=columns depth=32

    // Overlap patch generation and matrix multiplication.
    #pragma HLS DATAFLOW

    Im2col(input, columns);

    for (int position = 0; position < OUTPUT_VALUES; position++) {
        #pragma HLS PIPELINE II=1

        accumulator_t sum = 0;

        // Multiply the nine patch values by the nine weights.
        for (int i = 0; i < KERNEL_SIZE; i++) {
            #pragma HLS UNROLL
            for (int j = 0; j < KERNEL_SIZE; j++) {
                #pragma HLS UNROLL

                int index = i * KERNEL_SIZE + j;
                sum += columns[index][position] * weights[i][j];
            }
        }

        // account bias after all multiplications.
        sum += bias;

        int row = position / OUTPUT_SIZE;
        int col = position % OUTPUT_SIZE;

        // Update ReLU negative results with zero.
        if (sum > 0) {
            output[row][col] = activation_t(sum);
        } else {
            output[row][col] = 0;
        }
    }
}
