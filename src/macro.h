// Copyright 2022 Welink Inc. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the License); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//
// Author: yaoguangchao@pyou.com

#pragma once

#include <stdio.h>

#define TIMEDIFF(s, e) ((e.tv_sec - s.tv_sec) * 1000.0 + (e.tv_usec - s.tv_usec) / 1000.0)

#define TRUNCATE(a) ((a) > 255.0 ? 255 : ((a) < 0 ? 0 : (rint((a)))))

#ifdef _USE_CONFIG1_
// The size of a CUDA 1-d block, e.g. for vector operations..
#define CU1DBLOCK 128
#define STACKSIZE 24
#define STACKBYTES 49152    // CU1DBLOCK * STACKSIZE * 16
#elif defined _USE_CONFIG2_
#define CU1DBLOCK 160
#define STACKSIZE 19
#define STACKBYTES 48640
#elif defined _USE_CONFIG3_
#define CU1DBLOCK 192
#define STACKSIZE 16
#define STACKBYTES 49152
#elif defined _USE_CONFIG4_
// occupancy probe: 224 threads, 2 blocks/SM at 16B entries (448 thr/SM)
#define CU1DBLOCK 224
#define STACKSIZE 13
#define STACKBYTES 46592
#elif defined _USE_CONFIG5_
// occupancy probe: 128 threads, 3 blocks/SM at 16B entries (384 thr/SM)
#define CU1DBLOCK 128
#define STACKSIZE 15
#define STACKBYTES 30720
#elif defined _USE_CONFIG6_
// Q(28)-capable with v4 (depth = 28 - 6 rows - 2): 128 threads, 2 blocks/SM
#define CU1DBLOCK 128
#define STACKSIZE 20
#define STACKBYTES 40960
#elif defined _USE_CONFIG7_
// Q(28)-capable with v4: 96 threads, 3 blocks/SM (288 thr/SM)
#define CU1DBLOCK 96
#define STACKSIZE 20
#define STACKBYTES 30720
#endif

// stringify helper so the per-config stack size can appear inside inline PTX
#define NQ_STR2(x) #x
#define NQ_STR(x) NQ_STR2(x)

// The size of edge of CUDA square block, e.g. for matrix operations.
#define CU2DBLOCK 16

#define CU_SAFE_CALL(fun)                                                                        \
    {                                                                                            \
        int ret;                                                                                 \
        if ((ret = (fun)) != 0) {                                                                \
            fprintf(stderr, "[%s:%s(%d)]cudaError_t %d:%s\n", __FILE__, __func__, __LINE__, ret, \
                    cudaGetErrorString((cudaError_t)ret));                                       \
            exit(-1);                                                                            \
        }                                                                                        \
        cudaDeviceSynchronize();                                                                 \
    }
