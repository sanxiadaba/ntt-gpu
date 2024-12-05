#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <time.h>
#include <stdint.h>

int main() {
    const char* input_file = "./input.txt";
    const char* output_file = "./output.txt";

    mpz_t num1, num2, result;
    mpz_init(num1);
    mpz_init(num2);
    mpz_init(result);

    // 动态分配缓冲区，避免栈溢出
    size_t buffer_size = 70 * 1024 * 1024; // 每行约 70MB
    char* buffer1 = (char*)malloc(buffer_size);
    char* buffer2 = (char*)malloc(buffer_size);

    if (!buffer1 || !buffer2) {
        fprintf(stderr, "内存分配失败\n");
        free(buffer1);
        free(buffer2);
        return 1;
    }

    // 读取文件
    FILE* input = fopen(input_file, "r");
    if (input == NULL) {
        perror("无法打开输入文件");
        free(buffer1);
        free(buffer2);
        return 1;
    }
    if (fgets(buffer1, buffer_size, input) == NULL ||
        fgets(buffer2, buffer_size, input) == NULL) {
        perror("读取文件失败");
        fclose(input);
        free(buffer1);
        free(buffer2);
        return 1;
    }
    fclose(input);

    // 初始化大数
    mpz_set_str(num1, buffer1, 10);
    mpz_set_str(num2, buffer2, 10);

    printf("数字 1 的位数: %zu\n", mpz_sizeinbase(num1, 10));
    printf("数字 2 的位数: %zu\n", mpz_sizeinbase(num2, 10));

    uint64_t start = clock();
    // 计算乘积
    mpz_mul(result, num1, num2);
    printf("time is %d\r\n", (clock() - start) * 1000 / CLOCKS_PER_SEC);

    // 将结果写入文件
    FILE* output = fopen(output_file, "w");
    if (output == NULL) {
        perror("无法打开输出文件");
        mpz_clears(num1, num2, result, NULL);
        free(buffer1);
        free(buffer2);
        return 1;
    }
    gmp_fprintf(output, "%Zd\n", result);
    fclose(output);

    // 清理资源
    mpz_clears(num1, num2, result, NULL);
    free(buffer1);
    free(buffer2);

    printf("计算完成，结果已写入 %s\n", output_file);
    return 0;
}
