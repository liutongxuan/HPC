#include <iostream>
#include <sys/time.h>
#include <xmmintrin.h>
#include <immintrin.h>

int m = 1024;
int n = 512;
int k = 2037;

float A[1024*2037];// [m,k]
float B[2037*512]; // [k,n]
float C[1024*512]; // [m,n]

void init() {
  for (int i = 0; i < 1024*2037; ++i) {
    A[i] = i % 7;
  }
  for (int i = 0; i < 2037*512; ++i) {
    B[i] = i % 6;
  }
  for (int i = 0; i < 1024*512; ++i) {
    C[i] = i % 5;
  }
}

void matmul_base() {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int p = 0; p < k; ++p) {
        C[i*n+j] = C[i*n+j] + A[i*k+p] * B[p*n+j];
      }
    }
  }
}

void matmul_unrolling_1x4() {
  for (int j = 0; j < n; j+=4) {
    for (int i = 0; i < m; ++i) {
      int c_index_0 = i*n + j;
      int c_index_1 = c_index_0 + 1;
      int c_index_2 = c_index_0 + 2;
      int c_index_3 = c_index_0 + 3;
      for (int p = 0; p < k; ++p) {
        int a_index = i*k+p;
        int b_index = p*n+j;
        C[c_index_0] = C[c_index_0] + A[a_index] * B[b_index];
        C[c_index_1] = C[c_index_1] + A[a_index] * B[b_index+1];
        C[c_index_2] = C[c_index_2] + A[a_index] * B[b_index+2];
        C[c_index_3] = C[c_index_3] + A[a_index] * B[b_index+3];
      }
    }
  }
}

void matmul_unrolling_1x4_register_index() {
  for (int j = 0; j < n; j+=4) {
    for (int i = 0; i < m; ++i) {
      register int c_index_0 = i*n + j;
      register int c_index_1 = c_index_0 + 1;
      register int c_index_2 = c_index_0 + 2;
      register int c_index_3 = c_index_0 + 3;

      register float c_index_val_0 = 0;
      register float c_index_val_1 = 0;
      register float c_index_val_2 = 0;
      register float c_index_val_3 = 0;

      for (int p = 0; p < k; ++p) {
        register int a_index = i*k+p;
        register int b_index = p*n+j;
        register float a_index_val = A[a_index];
        c_index_val_0 += a_index_val * B[b_index];
        c_index_val_1 += a_index_val * B[b_index+1];
        c_index_val_2 += a_index_val * B[b_index+2];
        c_index_val_3 += a_index_val * B[b_index+3];

      }
      C[c_index_0] = c_index_val_0;
      C[c_index_1] = c_index_val_1;
      C[c_index_2] = c_index_val_2;
      C[c_index_3] = c_index_val_3;
    }
  }
}

void matmul_unrolling_1x8_register_index() {
  for (int j = 0; j < n; j+=8) {
    for (int i = 0; i < m; ++i) {
      register int c_index_0 = i*n + j;
      register int c_index_1 = c_index_0 + 1;
      register int c_index_2 = c_index_0 + 2;
      register int c_index_3 = c_index_0 + 3;
      register int c_index_4 = c_index_0 + 4;
      register int c_index_5 = c_index_0 + 5;
      register int c_index_6 = c_index_0 + 6;
      register int c_index_7 = c_index_0 + 7;

      register float c_index_val_0 = 0;
      register float c_index_val_1 = 0;
      register float c_index_val_2 = 0;
      register float c_index_val_3 = 0;
      register float c_index_val_4 = 0;
      register float c_index_val_5 = 0;
      register float c_index_val_6 = 0;
      register float c_index_val_7 = 0;

      for (int p = 0; p < k; ++p) {
        register int a_index = i*k+p;
        register int b_index = p*n+j;
        register float a_index_val = A[a_index];
        c_index_val_0 += a_index_val * B[b_index];
        c_index_val_1 += a_index_val * B[b_index+1];
        c_index_val_2 += a_index_val * B[b_index+2];
        c_index_val_3 += a_index_val * B[b_index+3];
        c_index_val_4 += a_index_val * B[b_index+4];
        c_index_val_5 += a_index_val * B[b_index+5];
        c_index_val_6 += a_index_val * B[b_index+6];
        c_index_val_7 += a_index_val * B[b_index+7];
      }
      C[c_index_0] = c_index_val_0;
      C[c_index_1] = c_index_val_1;
      C[c_index_2] = c_index_val_2;
      C[c_index_3] = c_index_val_3;
      C[c_index_4] = c_index_val_4;
      C[c_index_5] = c_index_val_5;
      C[c_index_6] = c_index_val_6;
      C[c_index_7] = c_index_val_7;
    }
  }
}

void matmul_unrolling_4x4_register_index() {
  for (int j = 0; j < n; j+=4) {
    for (int i = 0; i < m; ++i) {
      register int c_index_0 = i*n + j;
      register int c_index_1 = c_index_0 + 1;
      register int c_index_2 = c_index_0 + 2;
      register int c_index_3 = c_index_0 + 3;

      register float c_index_val_0 = 0;
      register float c_index_val_1 = 0;
      register float c_index_val_2 = 0;
      register float c_index_val_3 = 0;

      register int a_index_base = i*k;
      for (int p = 0; p < k; p+=4) {
        register int a_index_0 = a_index_base+p;
        register int a_index_1 = a_index_0+1;
        register int a_index_2 = a_index_0+2;
        register int a_index_3 = a_index_0+3;
        register int b_index_0 = p*n+j;
        register int b_index_1 = b_index_0+n;
        register int b_index_2 = b_index_1+n;
        register int b_index_3 = b_index_2+n;
        register float a_index_val_0 = A[a_index_0];
        register float a_index_val_1 = A[a_index_1];
        register float a_index_val_2 = A[a_index_2];
        register float a_index_val_3 = A[a_index_3];

        c_index_val_0 += a_index_val_0 * B[b_index_0];
        c_index_val_1 += a_index_val_0 * B[b_index_0+1];
        c_index_val_2 += a_index_val_0 * B[b_index_0+2];
        c_index_val_3 += a_index_val_0 * B[b_index_0+3];
        
        c_index_val_0 += a_index_val_1 * B[b_index_1];
        c_index_val_1 += a_index_val_1 * B[b_index_1+1];
        c_index_val_2 += a_index_val_1 * B[b_index_1+2];
        c_index_val_3 += a_index_val_1 * B[b_index_1+3];
        
        c_index_val_0 += a_index_val_2 * B[b_index_2];
        c_index_val_1 += a_index_val_2 * B[b_index_2+1];
        c_index_val_2 += a_index_val_2 * B[b_index_2+2];
        c_index_val_3 += a_index_val_2 * B[b_index_2+3];

        c_index_val_0 += a_index_val_3 * B[b_index_3];
        c_index_val_1 += a_index_val_3 * B[b_index_3+1];
        c_index_val_2 += a_index_val_3 * B[b_index_3+2];
        c_index_val_3 += a_index_val_3 * B[b_index_3+3];
      }

      C[c_index_0] = c_index_val_0;
      C[c_index_1] = c_index_val_1;
      C[c_index_2] = c_index_val_2;
      C[c_index_3] = c_index_val_3;
    }
  }
}

void matmul_unrolling_1x4_register_index_sse() {
  for (int j = 0; j < n; j+=4) {
    for (int i = 0; i < m; ++i) {
      register int c_index_0 = i*n + j;
      __m128 c_val = _mm_loadu_ps(&C[c_index_0]);

      for (int p = 0; p < k; ++p) {
        register int b_index = p*n+j;
        register int a_index = i*k+p;

        __m128 b_val = _mm_loadu_ps(&B[b_index]);
        __m128 a_val = _mm_load1_ps(&A[a_index]);

        c_val = _mm_add_ps(_mm_mul_ps(a_val, b_val), c_val);
      }
      _mm_store_ps(&C[c_index_0], c_val);
    }
  }
}

void matmul_unrolling_1x4_register_index_aligned_sse() {
  for (int j = 0; j < n; j+=4) {
    for (int i = 0; i < m; ++i) {
      register int c_index_0 = i*n + j;
      __m128 c_val = _mm_load_ps(&C[c_index_0]);

      for (int p = 0; p < k; ++p) {
        register int b_index = p*n+j;
        register int a_index = i*k+p;

        __m128 b_val = _mm_load_ps(&B[b_index]);
        __m128 a_val = _mm_load1_ps(&A[a_index]);

        c_val = _mm_add_ps(_mm_mul_ps(a_val, b_val), c_val);
      }
      _mm_store_ps(&C[c_index_0], c_val);
    }
  }
}

void matmul_unrolling_1x8_register_index_avx256() {
  for (int j = 0; j < n; j+=8) {
    for (int i = 0; i < m; ++i) {
      register int c_index_0 = i*n + j;
      __m256 c_val = _mm256_loadu_ps(&C[c_index_0]);

      for (int p = 0; p < k; ++p) {
        register int b_index = p*n+j;
        register int a_index = i*k+p;

        __m256 b_val = _mm256_loadu_ps(&B[b_index]);
        __m256 a_val = _mm256_set1_ps(A[a_index]);

        c_val = _mm256_add_ps(_mm256_mul_ps(a_val, b_val), c_val);
      }
      _mm256_store_ps(&C[c_index_0], c_val);
    }
  }
}

/*
void matmul_unrolling_1x4_register_index_avx512() {
  for (int j = 0; j < n; j+=16) {
    for (int i = 0; i < m; ++i) {
      register int c_index_0 = i*n + j;
      __m512 c_val = _mm512_load_ps(&C[c_index_0]);

      for (int p = 0; p < k; ++p) {
        register int b_index = p*n+j;
        register int a_index = i*k+p;

        __m512 b_val = _mm512_load_ps(&B[b_index]);
        __m512 a_val = _mm512_set1_ps(A[a_index]);

        c_val = _mm512_add_ps(_mm512_mul_ps(a_val, b_val), c_val);
      }
      _mm512_store_ps(&C[c_index_0], c_val);
    }
  }
}*/

void matmul_unrolling_1x4_register_index_sse_pack() {
  for (int j = 0; j < n; j+=4) {
    for (int i = 0; i < m; ++i) {
      register int c_index_0 = i*n + j;
      __m128 c_val = _mm_loadu_ps(&C[c_index_0]);

      float tmp_a[k];
      for (int p = 0; p < k; ++p) {
        tmp_a[p] = A[i*k+p];
      }

      for (int p = 0; p < k; ++p) {
        register int b_index = p*n+j;
        __m128 b_val = _mm_loadu_ps(&B[b_index]);
        __m128 a_val = _mm_load1_ps(&tmp_a[p]);

        c_val = _mm_add_ps(_mm_mul_ps(a_val, b_val), c_val);
      }
      _mm_store_ps(&C[c_index_0], c_val);
    }
  }
}

int main() {
  init();
  timeval start;
  gettimeofday(&start, nullptr);
  matmul_base();
  timeval stop;
  gettimeofday(&stop, nullptr);
  std::cout << "matmul_base eclapse:"
            << ((stop.tv_sec- start.tv_sec)* 1000 *1000 + (stop.tv_usec- start.tv_usec))
            << "us"
            << std::endl;

  std::cout << "===============================" << std::endl;
  gettimeofday(&start, nullptr);
  matmul_unrolling_1x4();
  gettimeofday(&stop, nullptr);
  std::cout << "matmul_unrolling_1x4 eclapse:"
            << ((stop.tv_sec- start.tv_sec)* 1000 *1000 + (stop.tv_usec- start.tv_usec))
            << "us"
            << std::endl;
  
  std::cout << "===============================" << std::endl;
  gettimeofday(&start, nullptr);
  matmul_unrolling_1x4_register_index();
  gettimeofday(&stop, nullptr);
  std::cout << "matmul_unrolling_1x4_register_index eclapse:"
            << ((stop.tv_sec- start.tv_sec)* 1000 *1000 + (stop.tv_usec- start.tv_usec))
            << "us"
            << std::endl;
  
  std::cout << "===============================" << std::endl;
  gettimeofday(&start, nullptr);
  matmul_unrolling_4x4_register_index();
  gettimeofday(&stop, nullptr);
  std::cout << "matmul_unrolling_4x4_register_index eclapse:"
            << ((stop.tv_sec- start.tv_sec)* 1000 *1000 + (stop.tv_usec- start.tv_usec))
            << "us"
            << std::endl;
  
  std::cout << "===============================" << std::endl;
  gettimeofday(&start, nullptr);
  matmul_unrolling_1x4_register_index_sse();
  gettimeofday(&stop, nullptr);
  std::cout << "matmul_unrolling_1x4_register_index_sse eclapse:"
            << ((stop.tv_sec- start.tv_sec)* 1000 *1000 + (stop.tv_usec- start.tv_usec))
            << "us"
            << std::endl;
  
  std::cout << "===============================" << std::endl;
  gettimeofday(&start, nullptr);
  matmul_unrolling_1x4_register_index_aligned_sse();
  gettimeofday(&stop, nullptr);
  std::cout << "matmul_unrolling_1x4_register_index_aligned_sse eclapse:"
            << ((stop.tv_sec- start.tv_sec)* 1000 *1000 + (stop.tv_usec- start.tv_usec))
            << "us"
            << std::endl;
  
  std::cout << "===============================" << std::endl;
  gettimeofday(&start, nullptr);
  matmul_unrolling_1x8_register_index_avx256();
  gettimeofday(&stop, nullptr);
  std::cout << "matmul_unrolling_1x4_register_index_avx256 eclapse:"
            << ((stop.tv_sec- start.tv_sec)* 1000 *1000 + (stop.tv_usec- start.tv_usec))
            << "us"
            << std::endl;

  std::cout << "===============================" << std::endl;
  gettimeofday(&start, nullptr);
  matmul_unrolling_1x8_register_index();
  gettimeofday(&stop, nullptr);
  std::cout << "matmul_unrolling_1x8_register_index eclapse:"
            << ((stop.tv_sec- start.tv_sec)* 1000 *1000 + (stop.tv_usec- start.tv_usec))
            << "us"
            << std::endl;
  /*
  std::cout << "===============================" << std::endl;
  gettimeofday(&start, nullptr);
  matmul_unrolling_1x4_register_index_avx512();
  gettimeofday(&stop, nullptr);
  std::cout << "matmul_unrolling_1x4_register_index_avx512 eclapse:"
            << ((stop.tv_sec- start.tv_sec)* 1000 *1000 + (stop.tv_usec- start.tv_usec))
            << "us"
            << std::endl;*/
  
  std::cout << "===============================" << std::endl;
  gettimeofday(&start, nullptr);
  matmul_unrolling_1x4_register_index_sse_pack();
  gettimeofday(&stop, nullptr);
  std::cout << "matmul_unrolling_1x4_register_index_sse_pack eclapse:"
            << ((stop.tv_sec- start.tv_sec)* 1000 *1000 + (stop.tv_usec- start.tv_usec))
            << "us"
            << std::endl;
  return 0;
}
