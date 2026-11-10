// CPU reference implementation: multiply the same two-line input with GMP.
//
// Build (MSYS2 / MinGW, or WSL):
//     g++ -O2 gmp_calc.cpp -lgmp -o gmp_calc
//
// This program exists to cross-check ntt_gpu. It reads ./input.txt and writes
// ./output.txt by default, and accepts explicit paths like ntt_gpu does.

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Reads one line of any length, dropping the line ending and one trailing
// carriage return. Returns NULL at end of file or on allocation failure.
static char *read_line(FILE *file) {
    size_t capacity = 1u << 20;
    size_t length = 0;
    char *buffer = (char *)malloc(capacity);
    if (buffer == NULL) {
        return NULL;
    }
    buffer[0] = '\0';

    for (;;) {
        if (length + 1 >= capacity) {
            size_t next;
            char *grown;
            if (capacity > ((size_t)1 << 30)) {
                free(buffer);
                return NULL;
            }
            next = capacity * 2;
            grown = (char *)realloc(buffer, next);
            if (grown == NULL) {
                free(buffer);
                return NULL;
            }
            buffer = grown;
            capacity = next;
        }
        if (fgets(buffer + length, (int)(capacity - length), file) == NULL) {
            if (length == 0) {
                free(buffer);
                return NULL;
            }
            break;
        }
        length += strlen(buffer + length);
        if (length > 0 && buffer[length - 1] == '\n') {
            buffer[--length] = '\0';
            break;
        }
    }
    if (length > 0 && buffer[length - 1] == '\r') {
        buffer[--length] = '\0';
    }
    return buffer;
}

int main(int argc, char **argv) {
    const char *input_path = argc > 1 ? argv[1] : "./input.txt";
    const char *output_path = argc > 2 ? argv[2] : "./output.txt";

    FILE *input = fopen(input_path, "rb");
    char *first = NULL;
    char *second = NULL;
    mpz_t left;
    mpz_t right;
    mpz_t result;
    FILE *output = NULL;

    if (input == NULL) {
        perror(input_path);
        return 1;
    }
    first = read_line(input);
    second = read_line(input);
    fclose(input);
    if (first == NULL || second == NULL) {
        fprintf(stderr, "error: %s must contain two decimal lines\n", input_path);
        free(first);
        free(second);
        return 1;
    }

    mpz_inits(left, right, result, NULL);
    if (mpz_set_str(left, first, 10) != 0 || mpz_set_str(right, second, 10) != 0) {
        fprintf(stderr, "error: both lines must be decimal integers\n");
        mpz_clears(left, right, result, NULL);
        free(first);
        free(second);
        return 1;
    }

    printf("digits: %lu x %lu\n",
           (unsigned long)mpz_sizeinbase(left, 10),
           (unsigned long)mpz_sizeinbase(right, 10));

    {
        const clock_t begin = clock();
        double elapsed_ms;
        mpz_mul(result, left, right);
        elapsed_ms = 1000.0 * (double)(clock() - begin) / (double)CLOCKS_PER_SEC;
        printf("multiply: %.1f ms (CPU time)\n", elapsed_ms);
    }

    output = fopen(output_path, "w");
    if (output == NULL) {
        perror(output_path);
        mpz_clears(left, right, result, NULL);
        free(first);
        free(second);
        return 1;
    }
    gmp_fprintf(output, "%Zd\n", result);
    fclose(output);

    printf("wrote %s (%lu digits)\n",
           output_path,
           (unsigned long)mpz_sizeinbase(result, 10));

    mpz_clears(left, right, result, NULL);
    free(first);
    free(second);
    return 0;
}
