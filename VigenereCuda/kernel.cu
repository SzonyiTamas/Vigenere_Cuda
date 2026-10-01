#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <map>
#include <iostream>
#include <time.h> 

#define TEXT_N 50000
#define KEY_M 64
#define BLOCK_SIZE 512

__device__ char dev_TEXT[TEXT_N];
__device__ char dev_KEY[KEY_M];
__device__ char dev_ENCRYPTED[TEXT_N];

__device__ char dev_KEYSTREAM[TEXT_N];
__device__ int dev_VALID[KEY_M];

__device__ char vigenere_encrypt_char(char plaintext, char key) {
    int p = plaintext - 'A';
    int k = key - 'A';
    int c = (p + k) % 26;
    return (char)(c + 'A');
}

__device__ char keystream_shift(char plain, char cipher) {
    int p = plain - 'A';
    int c = cipher - 'A';
    int v = (c - p) % 26;
    if (v < 0) v += 26;
    return (char)v;
}

__global__ void KernelEncrypt(int key_len)
{
    __shared__ char sh_KEY[KEY_M];

    if (threadIdx.x < key_len) {
        sh_KEY[threadIdx.x] = dev_KEY[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < TEXT_N) {
        char key_char = sh_KEY[idx % key_len];
        dev_ENCRYPTED[idx] = vigenere_encrypt_char(dev_TEXT[idx], key_char);
    }
}

__global__ void KernelKeyStream()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < TEXT_N) {
        dev_KEYSTREAM[idx] = keystream_shift(dev_TEXT[idx], dev_ENCRYPTED[idx]);
    }
}


__global__ void KernelTestPeriods(int max_key_len)
{
    int L = blockIdx.x + 1;
    if (L > max_key_len) return;

    __shared__ unsigned char ref[KEY_M];
    if (threadIdx.x < L) {
        ref[threadIdx.x] = (unsigned char)dev_KEYSTREAM[threadIdx.x];
    }
    __syncthreads();

    int bad = 0;
    for (int i = L + threadIdx.x; i < TEXT_N; i += blockDim.x) {
        if ((unsigned char)dev_KEYSTREAM[i] != ref[i % L]) {
            bad = 1;
            break;
        }
    }

    __shared__ int sh_bad[BLOCK_SIZE];
    sh_bad[threadIdx.x] = bad;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sh_bad[threadIdx.x] |= sh_bad[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        dev_VALID[L - 1] = (sh_bad[0] == 0) ? 1 : 0;
    }
}

int main()
{
    srand((unsigned)time(NULL));

    std::string abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    std::map<char, int> CharToInt;
    std::map<int, char> IntToChar;

    for (int i = 0; i < (int)abc.length(); i++)
    {
        CharToInt.insert({ (char)abc[i], i });
        IntToChar.insert({ i, (char)abc[i] });
    }

    std::string TEXT = "";
    std::string KEY = "";

    for (int i = 0; i < TEXT_N; i++)
    {
        char ch = 'a' + rand() % 26;
        TEXT += ch;
    }

    int TRUE_KEYLEN = 10;
    for (int i = 0; i < TRUE_KEYLEN; i++)
    {
        char ch = 'a' + rand() % 26;
        KEY += ch;
    }

    char TEXTARRAY[TEXT_N];
    char KEY_ARRAY[KEY_M];
    char ENCRYPTED_ARRAY[TEXT_N];

    for (int i = 0; i < TEXT_N; i++)
    {
        TEXT[i] = (char)toupper(TEXT[i]);
        TEXTARRAY[i] = TEXT[i];
    }

    for (int i = 0; i < KEY_M; i++)
        KEY_ARRAY[i] = 'A';

    for (int i = 0; i < TRUE_KEYLEN; i++)
        KEY_ARRAY[i] = (char)toupper(KEY[i]);

    cudaMemcpyToSymbol(dev_KEY, KEY_ARRAY, KEY_M * sizeof(char));
    cudaMemcpyToSymbol(dev_TEXT, TEXTARRAY, TEXT_N * sizeof(char));

    cudaEvent_t startG, stopG;
    cudaEventCreate(&startG);
    cudaEventRecord(startG, 0);

    int block_count = (TEXT_N - 1) / BLOCK_SIZE + 1;
    KernelEncrypt << <block_count, BLOCK_SIZE >> > (TRUE_KEYLEN);

    cudaEventCreate(&stopG);
    cudaEventRecord(stopG, 0);
    cudaEventSynchronize(stopG);

    float tEncrypt = 0;
    cudaEventElapsedTime(&tEncrypt, startG, stopG);

    cudaMemcpyFromSymbol(ENCRYPTED_ARRAY, dev_ENCRYPTED, TEXT_N * sizeof(char));

    std::string ENCRYPTED = "";
    for (int i = 0; i < TEXT_N; i++)
        ENCRYPTED += ENCRYPTED_ARRAY[i];

    cudaEvent_t startCPU, stopCPU;
    cudaEventCreate(&startCPU);
    cudaEventRecord(startCPU, 0);

    char ks_cpu[TEXT_N];

    for (int i = 0; i < TEXT_N; i++)
    {
        int p = CharToInt.at(TEXT[i]);
        int c = CharToInt.at(ENCRYPTED[i]);
        int v = (c - p) % 26;
        if (v < 0) v += 26;
        ks_cpu[i] = (char)v;
    }

    int cpuBestL = -1;
    for (int L = 1; L <= KEY_M; L++)
    {
        int ok = 1;
        for (int i = L; i < TEXT_N; i++)
        {
            if (ks_cpu[i] != ks_cpu[i % L]) { ok = 0; break; }
        }
        if (ok == 1) { cpuBestL = L; break; }
    }

    std::string CPU_FOUND_KEY = "";
    if (cpuBestL > 0)
    {
        for (int i = 0; i < cpuBestL; i++)
            CPU_FOUND_KEY += (char)('A' + ks_cpu[i]);
    }

    cudaEventCreate(&stopCPU);
    cudaEventRecord(stopCPU, 0);
    cudaEventSynchronize(stopCPU);

    float tCPU = 0;
    cudaEventElapsedTime(&tCPU, startCPU, stopCPU);

    cudaEvent_t startGPUC, stopGPUC;
    cudaEventCreate(&startGPUC);
    cudaEventRecord(startGPUC, 0);

    KernelKeyStream << <block_count, BLOCK_SIZE >> > ();

    int valid_host[KEY_M];
    for (int i = 0; i < KEY_M; i++) valid_host[i] = 0;
    cudaMemcpyToSymbol(dev_VALID, valid_host, KEY_M * sizeof(int));

    KernelTestPeriods << <KEY_M, BLOCK_SIZE >> > (KEY_M);

    cudaMemcpyFromSymbol(valid_host, dev_VALID, KEY_M * sizeof(int));

    int gpuBestL = -1;
    for (int L = 1; L <= KEY_M; L++)
    {
        if (valid_host[L - 1] == 1) { gpuBestL = L; break; }
    }

    char ks_prefix[KEY_M];
    for (int i = 0; i < KEY_M; i++) ks_prefix[i] = 0;

    cudaMemcpyFromSymbol(ks_prefix, dev_KEYSTREAM, KEY_M * sizeof(char));

    std::string GPU_FOUND_KEY = "";
    if (gpuBestL > 0)
    {
        for (int i = 0; i < gpuBestL; i++)
            GPU_FOUND_KEY += (char)('A' + ks_prefix[i]);
    }

    cudaEventCreate(&stopGPUC);
    cudaEventRecord(stopGPUC, 0);
    cudaEventSynchronize(stopGPUC);

    float tGPUCrack = 0;
    cudaEventElapsedTime(&tGPUCrack, startGPUC, stopGPUC);

    std::string TRUE_KEY_UP = "";
    for (int i = 0; i < TRUE_KEYLEN; i++)
        TRUE_KEY_UP += (char)toupper(KEY[i]);

    std::cout << "TRUE KEY      : " << TRUE_KEY_UP << std::endl;
    std::cout << "CPU FOUND KEY : " << CPU_FOUND_KEY << "  (L=" << cpuBestL << ")" << std::endl;
    std::cout << "GPU FOUND KEY : " << GPU_FOUND_KEY << "  (L=" << gpuBestL << ")" << std::endl;

    std::cout << "GPU encrypt time: " << tEncrypt << " ms" << std::endl;
    std::cout << "CPU crack time  : " << tCPU << " ms" << std::endl;
    std::cout << "GPU crack time  : " << tGPUCrack << " ms" << std::endl;

    return 0;
}