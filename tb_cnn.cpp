#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include "header.h"
#include "weights.h"

using namespace std;

// Check Im2col and compare HLS against direct convolution.
// Double is used only for the independent testbench reference.
int checkDesign(input_t input[INPUT_SIZE][INPUT_SIZE], string testName)
{
    input_t columns[FILTER_VALUES][OUTPUT_VALUES];
    activation_t output[OUTPUT_SIZE][OUTPUT_SIZE];
    int errors = 0;

    Im2col(input, columns);
    GemmConv2d0(input, output);

    for (int row = 0; row < OUTPUT_SIZE; row++)
    {
        for (int col = 0; col < OUTPUT_SIZE; col++)
        {
            double expected = 0;

            for (int i = 0; i < KERNEL_SIZE; i++)
            {
                for (int j = 0; j < KERNEL_SIZE; j++)
                {
                    int index = i * KERNEL_SIZE + j;
                    int position = row * OUTPUT_SIZE + col;

                    if (columns[index][position] != input[row + i][col + j])
                    {
                        errors++;
                    }

                    // Direct convolution: no Im2col in this calculation.
                    expected += input[row + i][col + j].to_double()
                                * weights[i][j].to_double();
                }
            }

            expected += bias.to_double();
            if (expected < 0)
            {
                expected = 0;
            }

            double actual = output[row][col].to_double();
            if (fabs(actual - expected) > 0.0000001)
            {
                if (errors < 5)
                {
                    cerr << "Mismatch at (" << row << ", " << col << ")"
                         << ": expected " << expected
                         << ", got " << actual << endl;
                }
                errors++;
            }
        }
    }

    cout << testName << ": " << errors << " errors" << endl;
    return errors;
}

// Test simple patterns, filter positions, and image boundaries.
int runSimpleTests()
{
    input_t input[INPUT_SIZE][INPUT_SIZE];
    int errors = 0;
    string names[5] =
    {
        "Zeros",
        "Ones",
        "Checkerboard",
        "Ramp",
        "Mixed pixels"
    };

    for (int test = 0; test < 5; test++)
    {
        for (int row = 0; row < INPUT_SIZE; row++)
        {
            for (int col = 0; col < INPUT_SIZE; col++)
            {
                double value = 0;

                if (test == 1)
                {
                    value = 1;
                }
                else if (test == 2)
                {
                    value = (row + col) % 2;
                }
                else if (test == 3)
                {
                    value = (row * INPUT_SIZE + col) / 783.0;
                }
                else if (test == 4)
                {
                    value = ((row * 17 + col * 13) % 256) / 255.0;
                }

                input[row][col] = value;
            }
        }
        errors += checkDesign(input, names[test]);
    }

    // Coordinates of single pixels used in the impulse tests.
    int testRows[13] =
    {
        10, 10, 10, 11, 11, 11, 12, 12, 12, 0, 0, 27, 27
    };

    int testCols[13] =
    {
        10, 11, 12, 10, 11, 12, 10, 11, 12, 0, 27, 0, 27
    };

    for (int test = 0; test < 13; test++)
    {
        for (int row = 0; row < INPUT_SIZE; row++)
        {
            for (int col = 0; col < INPUT_SIZE; col++)
            {
                input[row][col] = 0;
            }
        }
        input[testRows[test]][testCols[test]] = 1;
        errors += checkDesign(input, "Single pixel");
    }

    return errors;
}

// Read one notebook image and its post-ReLU reference output.
int runNotebookTest()
{
    ifstream imageFile("image.dat");
    ifstream traceFile("trace.dat");

    if (!imageFile || !traceFile)
    {
        cerr << "Cannot open image.dat or trace.dat." << endl;
        cerr << "Export both from the same model as weights.h." << endl;
        return 1;
    }

    input_t input[INPUT_SIZE][INPUT_SIZE];
    activation_t output[OUTPUT_SIZE][OUTPUT_SIZE];
    double beforeRelu[OUTPUT_VALUES];
    double reference[OUTPUT_VALUES];

    for (int row = 0; row < INPUT_SIZE; row++)
    {
        for (int col = 0; col < INPUT_SIZE; col++)
        {
            double pixel;
            if (!(imageFile >> pixel) || !isfinite(pixel) || pixel < 0 || pixel > 1)
            {
                cerr << "image.dat must contain 784 pixels from 0 to 1." << endl;
                return 1;
            }
            input[row][col] = pixel;
        }
    }

    string extra;
    if (imageFile >> extra)
    {
        cerr << "Extra data found after the 784 input pixels." << endl;
        return 1;
    }

    // First 676 trace values: convolution plus bias, before ReLU.
    for (int i = 0; i < OUTPUT_VALUES; i++)
    {
        if (!(traceFile >> beforeRelu[i]) || !isfinite(beforeRelu[i]))
        {
            cerr << "Invalid first convolution block in trace.dat." << endl;
            return 1;
        }
    }

    // Next 676 trace values: convolution output after ReLU.
    for (int i = 0; i < OUTPUT_VALUES; i++)
    {
        if (!(traceFile >> reference[i]) || !isfinite(reference[i]))
        {
            cerr << "Invalid ReLU block in trace.dat." << endl;
            return 1;
        }

        double expected = beforeRelu[i];
        if (expected < 0)
        {
            expected = 0;
        }
        if (fabs(reference[i] - expected) > 0.00001)
        {
            cerr << "The two convolution trace blocks do not match." << endl;
            return 1;
        }
    }

    int errors = checkDesign(input, "Notebook image / fixed-point reference");
    GemmConv2d0(input, output);

    // Allow for fixed-point truncation of normalized pixels and parameters.
    // Input step = 1/256. Weight and bias step = 1/4096.
    // Bound = sum(abs(quantized weights))/256 + 9/4096 + 1/4096.
    // Add 1e-5 to allow small floating-point reference differences.
    double tolerance = 10.0 / 4096.0 + 0.00001;
    for (int i = 0; i < KERNEL_SIZE; i++)
    {
        for (int j = 0; j < KERNEL_SIZE; j++)
        {
            tolerance += fabs(weights[i][j].to_double()) / 256.0;
        }
    }

    ofstream resultFile("comparison.csv");
    if (!resultFile)
    {
        cerr << "Cannot create comparison.csv." << endl;
        return 1;
    }

    resultFile << "row,col,notebook,hls,error,tolerance" << endl;
    resultFile << setprecision(12);
    double maxError = 0;
    double totalError = 0;

    for (int row = 0; row < OUTPUT_SIZE; row++)
    {
        for (int col = 0; col < OUTPUT_SIZE; col++)
        {
            int position = row * OUTPUT_SIZE + col;
            double actual = output[row][col].to_double();
            double error = fabs(actual - reference[position]);

            totalError += error;
            if (error > maxError)
            {
                maxError = error;
            }
            if (error > tolerance)
            {
                if (errors < 5)
                {
                    cerr << "Notebook mismatch at (" << row << ", " << col << ")"
                         << ": error = " << error << endl;
                }
                errors++;
            }

            resultFile << row << "," << col << "," << reference[position]
                       << "," << actual << "," << error << "," << tolerance << endl;
        }
    }

    cout << "Notebook tolerance: " << tolerance << endl;
    cout << "Maximum error: " << maxError << endl;
    cout << "Mean error: " << totalError / OUTPUT_VALUES << endl;
    cout << "Saved comparison.csv" << endl;
    return errors;
}

int main()
{
    cout << "Starting convolution tests..." << endl;

    int errors = runSimpleTests();
    errors += runNotebookTest();

    if (errors == 0)
    {
        cout << "PASS: All tests passed." << endl;
        return 0;
    }
    else
    {
        cout << "FAIL: " << errors << " errors found." << endl;
        return 1;
    }
}
