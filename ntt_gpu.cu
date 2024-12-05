#define _CRT_SECURE_NO_WARNINGS


// 输入文件
#define Z_INPUT_FILE "./input.txt"
#define Z_OUT_FILE "./output.txt"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <string>

// 每个block大小
dim3 block(256);

FILE* file = NULL;

typedef uint32_t z_data;
// 最开始读取数据开辟的数组大小
z_data N = 0;
// 模数
const z_data G = 3;
// 对应的原根
const z_data P = 2281701377;

// 数据是十进制
#define Y_N 1

// 取两个值最大
#define max(a, b) ((a) > (b) ? (a) : (b))


// 下文会用到的一些变量
z_data* C = NULL;
z_data* A_dev = NULL, * B_dev = NULL, * C_dev = NULL;
char* a = NULL, * b = NULL;
char* a_dev = NULL, * b_dev = NULL;
z_data  lim = 1, len = 0;
z_data* r_dev = NULL;
z_data* i_arr_dev = NULL;
z_data* z_tmp_dev = NULL;


// 读取数据
void read_line(uint8_t buf_01[], uint8_t buf_02[], char* path)
{
	FILE* fp;
	int line_len = 0; // 文件每行的长度

	// 打开文件
	fp = fopen(path, "r");
	// 打开失败
	if (NULL == fp)
	{
		printf("open %s failed.\n", path);
		return;
	}

	// 读取第一行
	fgets((char*)buf_01, N, fp);
	line_len = (int)strlen((char*)buf_01);
	// 排除换行符
	if ('\n' == buf_01[line_len - 1])
	{
		buf_01[line_len - 1] = '\0';
		line_len--;
	}
	// windos文本排除回车符
	if ('\r' == buf_01[line_len - 1])
	{
		buf_01[line_len - 1] = '\0';
		line_len--;
	}

	// 读取第二行
	fgets((char*)buf_02, N, fp);
	line_len = (int)strlen((char*)buf_02);
	// 排除换行符
	if ('\n' == buf_02[line_len - 1])
	{
		buf_02[line_len - 1] = '\0';
		line_len--;
	}
	// windos文本排除回车符
	if ('\r' == buf_02[line_len - 1])
	{
		buf_02[line_len - 1] = '\0';
		line_len--;
	}

	//	printf("%s\r\n", buf_01);
	//	printf("%s\r\n", buf_02);

	fclose(fp);
}

const uint64_t z_one = 1;
// CPU 快速幂
z_data qpow(z_data x, z_data y)
{
	z_data res = 1;
	while (y)
	{
		if (y & 1)
		{
			res = z_one * res * x % P;
		}
		x = z_one * x * x % P;
		y >>= 1;
	}
	return (z_data)res;
}

// 数据初始化
__global__
void z_cuda_00(char data1[], char data2[], z_data m_data1[], z_data m_data2[], z_data len1, z_data len2, z_data y_n1, z_data y_n2)
{
	z_data i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len1 / Y_N) {
		z_data i_t = y_n1 + (len1 / Y_N - i) * Y_N;
		z_data m_t = 1;
		z_data data = 0;
		for (int32_t j = 1; j < Y_N + 1; j++)
		{
			data = (data1[i_t - j] - '0') * m_t + data;
			m_t = m_t * 10;
		}
		m_data1[i] = data;
	}
	if (i < len2 / Y_N) {
		z_data i_t = y_n2 + (len2 / Y_N - i) * Y_N;
		z_data m_t = 1;
		z_data data = 0;
		for (int32_t j = 1; j < Y_N + 1; j++)
		{
			data = (data2[i_t - j] - '0') * m_t + data;
			m_t = m_t * 10;
		}
		m_data2[i] = data;
	}
}

// 位倒序数组的生成
__global__
void z_cuda_01(z_data r[], z_data lim, z_data step)
{
	z_data i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < lim && (i & step) == step)
	{
		r[i] = (i & 1) * (lim >> 1) + (r[i >> 1] >> 1);
	}
}


// 位倒序
// 对缓存命中很不友好，性能很低
__global__
void z_cuda_02(z_data data[], z_data lim, z_data r[])
{
	z_data i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < lim && r[i] < i)
	{
		z_data tmp = data[i];
		z_data r_i = r[i];
		data[i] = data[r_i];
		data[r_i] = tmp;
	}
}


// 核心NTT代码
__global__
void z_cuda_03(z_data x[], z_data z_tmp[], z_data k, z_data P, z_data i_arr[], z_data all_n)
{
	z_data id = blockIdx.x * blockDim.x + threadIdx.x;
	z_data i = id / k;
	z_data j = id % k;
	z_data i_arr_i = i_arr[i];
	if (id < all_n)
	{
		z_data tmp = z_one * x[i_arr_i + j + k] * z_tmp[j] % P;
		x[i_arr_i + j + k] = (z_one * x[i_arr_i + j] + (P - tmp)) % P;
		x[i_arr_i + j] = (z_one * x[i_arr_i + j] + tmp) % P;
	}
}


// 反转数组
__global__
void z_cuda_04(z_data data[], z_data begin, z_data end, z_data mid)
{
	z_data i = blockIdx.x * blockDim.x + threadIdx.x;

	if (0 < i && i < mid)
	{
		z_data tmp = data[i];
		data[i] = data[end + 1 - i];
		data[end + 1 - i] = tmp;
	}
}

// 逆变换用到的一步处理
__global__
void z_cuda_05(z_data data[], z_data lim, z_data inv)
{
	z_data i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < lim)
	{
		data[i] = z_one * data[i] * inv % P;
	}
}

// 两个输入数组点相乘
__global__
void z_cuda_06(z_data A[], z_data B[], z_data C[], z_data lim, z_data P)
{
	z_data i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < lim)
	{
		C[i] = z_one * A[i] * B[i] % P;
	}
}

// 一个辅助处理函数
__global__
void z_cuda_07(z_data i_arr[], z_data size, z_data m)
{
	z_data i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size)
	{
		i_arr[i] = i * m;
	}
}

// 对一些中间变量的预处理
void i_arr_pro()
{
	z_data i_arr_index = 0;
	for (z_data m = 2; m <= lim; m <<= 1)
	{
		z_data i_num = lim / m;
		dim3 grid((i_num + block.x - 1) / block.x);
		z_cuda_07 << <grid, block >> > (i_arr_dev + i_arr_index, i_num, m);
		i_arr_index += i_num;
	}
}

// gpu快速幂
__device__
z_data qpow_gpu(z_data x, z_data y)
{
	z_data res = 1;
	while (y)
	{
		if (y & 1)
		{
			res = z_one * res * x % P;
		}
		x = z_one * x * x % P;
		y >>= 1;
	}
	return res;
}

// 对一些中间变量的预处理
__global__
void num_arr_pro(z_data num_arr[], z_data num, z_data n, z_data gn)
{
	z_data i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num)
	{
		num_arr[i] = qpow_gpu(gn, i * n);
	}
}

// 对一些中间变量的预处理
__global__ void gpu_pro(z_data gpu_arr[], z_data num_arr[], z_data n, z_data gn, z_data num)
{
	z_data i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num)
	{
		return;
	}
	gpu_arr[i * n] = num_arr[i];
	i = i * n;
	for (z_data j = 1; j < n; j++)
	{
		gpu_arr[i + j] = z_one * gpu_arr[i + j - 1] * gn % P;
	}
}

#define min(a,b) ((a) < (b) ? (a) : (b))

// 对一些中间变量的预处理
void z_tmp_pro_gpu()
{
	z_data m_num = 0;
	for (z_data m = 2; m <= lim; m <<= 1)
	{
		m_num++;
	}

	z_data* gn_arr = (z_data*)malloc(sizeof(z_data) * m_num);
	z_data* k_arr = (z_data*)malloc(sizeof(z_data) * m_num);
	z_data* k_arr_sum = (z_data*)malloc(sizeof(z_data) * (m_num + 1));

	k_arr_sum[0] = 0;
	z_data k_arr_sum_index = 1;
	z_data tmp_index = 0;
	for (z_data m = 2; m <= lim; m <<= 1)
	{
		gn_arr[tmp_index] = qpow(G, (P - 1) / m);
		k_arr[tmp_index] = m >> 1;
		k_arr_sum[k_arr_sum_index] = k_arr_sum[k_arr_sum_index - 1] + k_arr[tmp_index];
		tmp_index++;
		k_arr_sum_index++;
	}

	// ============
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	// 最大线程数
	z_data max_num = devProp.multiProcessorCount * devProp.maxThreadsPerBlock;
	z_data num = 1;
	while (num <= max_num)
	{
		num <<= 1;
	}
	num >>= 1;
	num = 1024 * 8;
	z_data tmp_num = num;
	// gpu分成了num个迭代节点
	z_data* num_arr = NULL;
	cudaMalloc((void**)&num_arr, num * sizeof(z_data));
	// ===========

	for (tmp_index = 0; tmp_index < m_num; tmp_index++)
	{
		z_data k = k_arr[tmp_index];
		z_data gn = gn_arr[tmp_index];
		z_data k_arr_sum_index_tmp = k_arr_sum[tmp_index];

		num = min(k, tmp_num);
		// 每个迭代节点迭代多少次
		z_data n = k / num;
		dim3 grid(((z_data)num + block.x - 1) / block.x);

		num_arr_pro << <grid, block >> > (num_arr, num, n, gn);
		gpu_pro << <grid, block >> > (z_tmp_dev + k_arr_sum_index_tmp, num_arr, n, gn, num);
	}

	free(gn_arr);
	free(k_arr);
}



// 预处理
void pre_pro(void)
{
	i_arr_pro();
	z_tmp_pro_gpu();
}


// ntt变换
void ntt(z_data* x_dev, int opt)
{
	z_data z_tmp_dev_index = 0;
	z_data i_arr_dev_index = 0;
	for (z_data m = 2; m <= lim; m <<= 1)
	{
		z_data k = m >> 1;
		z_data i_num = lim / m;
		z_data all_n = i_num * k;

		dim3 grid(((z_data)all_n + block.x - 1) / block.x);
		z_cuda_03 << <grid, block >> > (x_dev, z_tmp_dev + z_tmp_dev_index, k, P, i_arr_dev + i_arr_dev_index, all_n);
		z_tmp_dev_index += k;
		i_arr_dev_index += i_num;
	}

	// ntt逆变换
	if (opt == -1)
	{
		dim3 grid(((z_data)lim + block.x - 1) / block.x);
		z_cuda_04 << <grid, block >> > (x_dev, 1, lim - 1, lim / 2);
		z_data inv = qpow(lim, P - 2);
		z_cuda_05 << <grid, block >> > (x_dev, lim, inv);
	}
}


// 初始化
void init()
{
	C = (z_data*)calloc(lim, sizeof(z_data));

	cudaMalloc((void**)&A_dev, lim * sizeof(z_data));
	cudaMalloc((void**)&B_dev, lim * sizeof(z_data));
	cudaMalloc((void**)&C_dev, lim * sizeof(z_data));

	cudaMalloc((void**)&r_dev, lim * sizeof(z_data));
	cudaMalloc((void**)&z_tmp_dev, lim * sizeof(z_data));
	cudaMalloc((void**)&i_arr_dev, lim * sizeof(z_data));
}

// 函数实现：返回大于等于n的最小2的次方
unsigned int nextPowerOfTwo(unsigned int n)
{
	if (n == 0) return 1;  // 特殊情况：0的下一个2的次方是1

	// 如果n已经是2的次方，则直接返回n
	if ((n & (n - 1)) == 0) return n;

	// 计算最小的2的次方
	return (1 << static_cast<int>(std::log2(n) + 1));
}


void read()
{
	uint64_t max_count = 0;
	std::ifstream file(Z_INPUT_FILE);  // 打开文件
	if (!file.is_open())
	{
		std::cerr << "无法打开文件" << std::endl;
		exit(-1);
	}

	std::string line;
	int lineCount = 0;

	// 按行读取文件
	while (std::getline(file, line))
	{
		lineCount++;
		// 输出每一行的列数，不包括换行符
		max_count = (max_count) > (line.length()) ? (max_count) : (line.length());
	}
	N = nextPowerOfTwo(max_count);
	file.close();  // 关闭文件

	a = (char*)calloc(N, sizeof(char));
	b = (char*)calloc(N, sizeof(char));

	cudaMalloc((void**)&a_dev, N * sizeof(char));
	cudaMalloc((void**)&b_dev, N * sizeof(char));

	read_line((uint8_t*)a, (uint8_t*)b, (char*)(Z_INPUT_FILE));

	z_data n1 = (z_data)strlen(a);
	z_data n2 = (z_data)strlen(b);

	z_data n1_t = n1 / Y_N + 1;
	z_data n2_t = n2 / Y_N + 1;

	while (lim < (n1_t << 1)) lim <<= 1;
	while (lim < (n2_t << 1)) lim <<= 1;

	//printf("lim is %d\r\n", lim);

	init();

	cudaMemcpy(a_dev, a, n1 * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(b_dev, b, n2 * sizeof(char), cudaMemcpyHostToDevice);

	z_data y_n1 = n1 % Y_N;
	z_data y_n2 = n2 % Y_N;

	dim3 grid(((z_data)lim + block.x - 1) / block.x);
	z_cuda_00 << <grid, block >> > (a_dev, b_dev, A_dev, B_dev, n1, n2, y_n1, y_n2);
	//printf("y_n1 is %d,n1 / Y_N is %d\r\n", y_n1, n1 / Y_N);
	if (y_n1 != 0)
	{
		z_data A_first[1] = { 0 };
		z_data data_t = 0;
		z_data j = 1;
		for (int32_t i = y_n1 - 1; i >= 0; i--)
		{
			data_t = (a[i] - '0') * j + data_t;
			j = j * 10;
		}
		A_first[0] = data_t;
		//printf("data_t is %d\r\n", data_t);
		cudaMemcpy(A_dev + (n1 / Y_N), A_first, sizeof(z_data), cudaMemcpyHostToDevice);
	}
	if (y_n2 != 0)
	{
		z_data B_first[1] = { 0 };
		z_data data_t = 0;
		z_data j = 1;
		for (int32_t i = y_n2 - 1; i >= 0; i--)
		{
			data_t = (b[i] - '0') * j + data_t;
			j = j * 10;
		}
		B_first[0] = data_t;
		cudaMemcpy(B_dev + (n2 / Y_N), B_first, sizeof(z_data), cudaMemcpyHostToDevice);
	}
	//uint32_t aaa[8] = {0};
	//cudaMemcpy(aaa, A_dev, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	//printf("hello\r\n");
	//for (size_t i = 0; i < 8; i++)
	//{
	//	printf("%d ", aaa[i]);
	//}

	//printf("\r\n");
	free(a);
	free(b);
	cudaFree(a_dev);
	cudaFree(b_dev);
}




void work()
{
	dim3 grid(((z_data)lim + block.x - 1) / block.x);
	for (z_data step = 1; step < lim; step *= 2)
	{
		z_cuda_01 << <grid, block >> > (r_dev, lim, step);
	}

	pre_pro();

	z_cuda_02 << <grid, block >> > (A_dev, lim, r_dev);
	z_cuda_02 << <grid, block >> > (B_dev, lim, r_dev);
	ntt(A_dev, 1);
	ntt(B_dev, 1);
	z_cuda_06 << <grid, block >> > (A_dev, B_dev, C_dev, lim, P);
	z_cuda_02 << <grid, block >> > (C_dev, lim, r_dev);
	ntt(C_dev, -1);
}

void print()
{
	cudaMemcpy(C, C_dev, lim * sizeof(z_data), cudaMemcpyDeviceToHost);
	z_data num = 1;
	for (z_data i = 0; i < Y_N; i++)
	{
		num = num * 10;
	}
	for (z_data i = 0; i < lim; ++i)
	{
		if (C[i] >= num) len = i + 1, C[i + 1] += C[i] / num, C[i] %= num;
		if (C[i]) len = max(len, i);
	}
	while (C[len] >= num) C[len + 1] += C[len] / num, C[len] %= num, len++;
	//for (uint32_t i = len; ~i; --i)
	//{
	//	fprintf(file, "%d", C[i]);
	//}

	for (int32_t i = len; i >= 0; --i)  // 注意：len 是从最高有效位开始
	{
		if (i == len) {
			// 打印最高位（不需要补零）
			fprintf(file, "%d", C[i]);
		}
		else {
			// 打印其他位，补零输出
			fprintf(file, "%0*u", Y_N, C[i]);
		}
	}
	fprintf(file, "\n");
}

void z_free()
{
	free(C);
	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);
	cudaFree(r_dev);
	cudaFree(i_arr_dev);
	cudaFree(z_tmp_dev);
}

#include <time.h>

int main()
{
	file = fopen(Z_OUT_FILE, "w");

	if (file == NULL)
	{
		exit(-1);
	}

	read();
	uint64_t start = clock();
	work();
	cudaDeviceSynchronize(); // 等待 GPU 完成任务
	printf("time is %dms\r\n", (clock() - start) * 1000 / CLOCKS_PER_SEC);
	print();
	z_free();
	return 0;
}
