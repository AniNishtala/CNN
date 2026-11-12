#ifndef CNN_H
#define CNN_H

#include "ap_fixed.h"

//CNN dimensions 
#define INPUT_SIZE 28
#define KERNEL_SIZE 3
#define OUTPUT_SIZE 26
#define FILTER_VALUES 9
#define OUTPUT_VALUES 676

// Normalized input pixel from 0 to 1.
typedef ap_ufixed<9, 1> input_t;

// weights + bias stored in weights.h.
typedef ap_fixed<16, 4> weight_t;
typedef ap_fixed<16, 4> bias_t;

// Sum of nine products + bias.
typedef ap_fixed<32, 8> accumulator_t;

// Output after ReLU.
typedef ap_ufixed<32, 8> activation_t;

// Im2Col funct. Transform input patches into a 9 x 676 matrix.
void Im2col
(
    input_t input[INPUT_SIZE][INPUT_SIZE],
    input_t columns[FILTER_VALUES][OUTPUT_VALUES]
);

// get the fixed weight from weights.h just like prev lab
void GemmConv2d0
(
    input_t input[INPUT_SIZE][INPUT_SIZE],
    activation_t output[OUTPUT_SIZE][OUTPUT_SIZE]
);

#endif