/****************************************************************************
 * Copyright (C) 2019 by Soulaymen Chouri                                   *
 *                                                                          *
 * This file is part of TensorBolt.                                         *
 *                                                                          *
 * What follows is the Modified BSD License.                                *
 *     See also http://www.opensource.org/licenses/BSD-3-Clause             *
 * Copyright (c) 2019, Soulaymen Chouri. All rights reserved.               *
 * Redistribution and use in source and binary forms, with or without       *
 * modification, are permitted provided that the following conditions       *
 * are met:                                                                 *
 *                                                                          *
 *      1. Redistributions of source code must retain the above copyright   *
 *         notice, this list of conditions and the following disclaimer.    *
 *                                                                          *
 *      2. Redistributions in binary form must reproduce the above          *
 *         copyright notice, this list of conditions and the following      *
 *         disclaimer in the documentation and/or other materials provided  *
 *         with the distribution.                                           *
 *                                                                          *
 *      3. Neither the name of the author nor the names of other            *
 *         contributors may be used to endorse or promote products derived  *
 *         from this software without specific prior written permission.    *
 *                                                                          *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS "AS IS" AND ANY EXPRESS OR      *
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED           *
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE   *
 * DISCLAIMED. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT,      *
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES       *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR       *
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)       *
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,      *
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING    *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE       *
 * POSSIBILITY OF SUCH DAMAGE.                                              *
 ****************************************************************************/

/**
 * @file tb_tensor.c
 * @author Soulaymen Chouri
 * @date March 16 2019
 * @brief File containing Tensors metadata & implementations.
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>

#include "ndarray.h"
#include "ndarray_std.h"

#define ranf()   ((rand())/(double)RAND_MAX)

/*
 The following function comes from this source:
 ftp://ftp.taygeta.com/pub/c/boxmuller.c
 */

/* boxmuller.c           Implements the Polar form of the Box-Muller
                         Transformation
 
                         (c) Copyright 1994, Everett F. Carter Jr.
                         Permission is granted by the author to use
                         this software for any application provided this
                         copyright notice is preserved.
 */

float box_muller(float m, float s)    /* normal random variate generator */
{                                     /* mean m, standard deviation s */
    float x1, x2, w, y1;
    static float y2;
    static int use_last = 0;
    
    if (use_last)                     /* use value from previous call */
    {
        y1 = y2;
        use_last = 0;
    }
    else
    {
        do {
            x1 = 2.0 * ranf() - 1.0;
            x2 = 2.0 * ranf() - 1.0;
            w = x1 * x1 + x2 * x2;
        } while ( w >= 1.0 );
        
        w = sqrt( (-2.0 * logf( w ) ) / w );
        y1 = x1 * w;
        y2 = x2 * w;
        use_last = 1;
    }
    
    return( m + y1 * s );
}



void nda_assert(int cond, const char * rawcond, const char* func_name, const char * fmt, ...){
    if(cond)
        return;
    char temp[1024];
    va_list vl;
    va_start(vl, fmt);
    vsprintf(temp, fmt, vl);
    va_end(vl);
    fprintf(stdout, "Fatal error, assertion failed: `%s` in function `%s` \n", rawcond, func_name);
    fprintf(stdout, "%s", temp);
    fprintf(stdout, "\n");
    assert(cond);
    exit(-1);
}
static inline uint64_t* __nda__copyArray(uint64_t len, uint64_t* array){
    uint64_t* arr = calloc(len, sizeof(uint64_t));
    memcpy(arr, array, sizeof(uint64_t)*len);
    return arr;
}


// https://stackoverflow.com/questions/29142417/4d-position-from-1d-index
void _printSubNDArrayRecursive(NDShape* shape, uint64_t* stack, uint64_t stack_len, uint64_t stack_value, uint64_t index, tb_float* data);


void nda_ShapeStackInit(NDShapeStack* stack, NDShape* shape){
    stack->shape = shape;
    stack->i = shape->rank;
}

uint8_t nda_ShapeStackCanPop(NDShapeStack* stack){
    return stack->i > 0;
}

uint64_t nda_ShapeStackPop(NDShapeStack* stack){
    stack->i--;
    return stack->shape->dims[stack->i];
}

NDShape* nda_newShape(uint64_t rank, ...){
    NDShape* shape = calloc(1, sizeof(NDShape));
    shape->rank = rank;
    shape->dims = calloc(rank, sizeof(uint64_t));
    size_t i = 0;
    va_list argPtr;
    va_start( argPtr, rank );
    for(; i < rank; i++ ){
        shape->dims[i] = va_arg( argPtr, uint64_t);
    }
    va_end( argPtr );
    
    shape->raw_len = nda_getTotalSize(shape);
    
    shape->strides = calloc(rank, sizeof(uint64_t));
    shape->strides[rank-1] = 1;
    
    for(i = rank-1; i > 0; i--){
        shape->strides[i-1] = shape->dims[i]*shape->strides[i];
    }

    return shape;
}

NDShape* nda_newShapeFromArray(uint64_t rank, uint64_t* dims){
    NDShape* shape = calloc(1, sizeof(NDShape));
    shape->rank = rank;
    shape->dims = dims;
    shape->raw_len = nda_getTotalSize(shape);
    
    shape->strides = calloc(rank, sizeof(uint64_t));
    shape->strides[rank-1] = 1;
    
    size_t i = 0;
    for(i = rank-1; i > 0; i--){
        shape->strides[i-1] = shape->dims[i]*shape->strides[i];
    }
    
    return shape;
}

NDShape* nda_newShapeFromArrayCopy(uint64_t rank, uint64_t* dims){
    uint64_t* arr = __nda__copyArray(rank, dims);
    return nda_newShapeFromArray(rank, arr);
}

void nda_debugShape(NDShape* shape){
    printf("Tensor{ .rank = %"PRIu64", \n\t", shape->rank);
    
    size_t i = 0;
    
    for(; i < shape->rank; i++){
        printf(".dims[%zu] = %"PRIu64", ", i, shape->dims[i]);
    }
    printf("\n\t");
    for(i=0; i < shape->rank; i++){
        printf(".stride[%zu] = %"PRIu64", ", i, shape->strides[i]);
    }

    printf("}\n");
}

char* nda_shapeToString(NDShape* shape){
    char buf[1024] = {0};
    int offset = snprintf(buf, 1024, "(.rank = %"PRIu64", ", shape->rank);
    
    size_t i = 0;
    for(; i < shape->rank; i++){
        offset += snprintf(buf+offset, 1024, ".dims[%zu] = %"PRIu64", ", i, shape->dims[i]);
    }
    
    snprintf(buf+offset, 1024, ")");
    
    return strdup(buf);
}

/**
 * \brief Returns the value of an array
 * \param[in] array NDArray to access
 * \param[in] index Array of dims of the element
 */
tb_float nda_get(NDArray* array, uint64_t* index){
    NDShape* shape = array->shape;
    
    uint64_t i = 0;
    uint64_t data_index = 0;
    
    for(; i < shape->rank; i++){
        ASSERT(index[i] < shape->dims[i], "Cannot index array with index %lld > dimension %lld in axis %lld", index[i], shape->dims[i], i);
        
        data_index += index[i]*shape->strides[i];
    }
    
    return array->data[data_index];
}


NDArray* nda_slice(struct NDArray* array, uint64_t* index){
    NDShape* shape = array->shape;
    uint64_t rank = shape->rank;
    
    uint64_t i = 0;
    uint64_t idx = 0;
    
    uint64_t new_rank = 0;
    
    for(; i < rank; i++, idx+=2){
        ASSERT(index[idx+1] >= index[idx], "Cannot get slice of range %lld:%lld", index[idx], index[idx+1]);
        if(index[idx+1] - index[idx] >= 1){
            new_rank++;
        }
    }
    
    NDShape* new_shape = calloc(1, sizeof(NDShape));
    new_shape->rank = new_rank;
    new_shape->dims = calloc(new_rank, sizeof(uint64_t));
    new_shape->strides = calloc(new_rank, sizeof(uint64_t));
    uint64_t padding = 0;
    i = 0;
    idx = 0;
    uint64_t counter = 0;
    for(; i < rank; i++, idx+=2){
        if(index[idx+1] - index[idx] > 0){
            new_shape->dims[counter] = index[idx+1] - index[idx];
            new_shape->strides[counter] = shape->strides[i];
            counter++;
        }
        
        padding += index[idx]*shape->strides[i];
    }
    
    new_shape->raw_len = nda_getTotalSize(new_shape);
    
    NDArray* arr = calloc(1, sizeof(NDArray));
    arr->shape = new_shape;
    arr->data = (tb_float*)(array->data+padding);
    
    return arr;
}

tb_float nda_vget(NDArray* array, uint64_t* index, NDShape* vshape){
    
    NDShape* shape = array->shape;
    
    uint64_t i = 0;
    int diff = vshape->rank-shape->rank;
    uint64_t data_index = 0;
    
    for(; i < shape->rank; i++){
        uint64_t id = index[i+diff];
        if (id >= shape->dims[i]){
            id %= shape->dims[i];
        }
        data_index += id*shape->strides[i];
    }
    
    return array->data[data_index];
}



tb_float nda_get1D(NDArray* array, uint64_t index){
    ASSERT(index < array->shape->raw_len, "Cannot fetch index %lld >= total size %lld of the array", index, array->shape->raw_len);
    
    return array->data[index];
}

tb_float nda_vget1D(NDArray* array, uint64_t index){
    while(index >= array->shape->raw_len)
        index = index/array->shape->raw_len;
    
    printf("fetching %lld\n", index);
    return array->data[index];
}

void nda_debugValue(NDArray* tensor){
    nda_debugShape(tensor->shape);
    NDShape* shape = tensor->shape;
    uint64_t len = tensor->shape->raw_len;
    uint64_t* idx = calloc(shape->rank, sizeof(uint64_t));
    
    size_t i = 0;
    size_t j = 0;
    for(; i < len; i++){
        uint64_t mod = 0;
        
        for(j = 0; j < shape->rank ;j++){
            idx[j] = (i - mod)/shape->strides[j] % shape->dims[j];
            mod = mod - idx[j]*shape->strides[j];
        }
        
        printf("\t%f,\n", nda_get(tensor, idx));
    }

    printf("\n");
    free(idx);
}

uint64_t nda_getTotalSize(NDShape* shape){
    uint64_t size = 1;

    size_t i = 0;

    for(; i < shape->rank; i++){
        size *= shape->dims[i];
    }

    return size;
}

NDShape* nda_copyShape(NDShape* shape){
    NDShape* shape2 = calloc(1, sizeof(NDShape));
    shape2->rank = shape->rank;
    shape2->dims = calloc(shape->rank, sizeof(uint64_t));
    memcpy(shape2->dims, shape->dims, shape->rank*sizeof(uint64_t));
    
    shape2->strides = calloc(shape->rank, sizeof(uint64_t));
    memcpy(shape2->strides, shape->strides, shape->rank*sizeof(uint64_t));
    
    shape2->raw_len = shape->raw_len;
    
    return shape2;
}

uint8_t nda_shapeCanBroadCast(NDShape* shape1, NDShape* shape2){
    NDShapeStack stack1, stack2;
    
    nda_ShapeStackInit(&stack1, shape1);
    nda_ShapeStackInit(&stack2, shape2);
    
    while(nda_ShapeStackCanPop(&stack1) && nda_ShapeStackCanPop(&stack2)){
        uint64_t i1 = nda_ShapeStackPop(&stack1);
        uint64_t i2 = nda_ShapeStackPop(&stack2);
        
        if((i1 != i2) && !((i1 != 1) || (i2 != 1))){
            return 0;
        }
    }
    
    if(shape1->raw_len > shape2->raw_len){
        return 1;
    }
    if(shape1->raw_len < shape2->raw_len){
        return 2;
    }
    
    if(shape1->rank > shape2->rank){
        return 1;
    }
    
    return 2;
}

NDArray* nda_alloc(NDShape* shape){
    uint64_t len = nda_getTotalSize(shape);
    tb_float* raw = calloc(len, sizeof(tb_float));

    NDArray* x = calloc(1, sizeof(NDArray));

    x->shape = shape;
    x->data = raw;

    return x;
}


NDArray* nda_randomNormal(struct NDShape* shape, float mu, float sig){
    uint64_t len = nda_getTotalSize(shape);
    tb_float* raw = calloc(len, sizeof(tb_float));
    
    NDArray* x = calloc(1, sizeof(NDArray));
    
    uint64_t i = 0;
    
    srand(time(NULL));
    
    for(; i < len; i++){
        raw[i] = (tb_float)box_muller(mu, sig);
    }
    
    x->shape = shape;
    x->data = raw;
    
    return x;
}

NDArray* nda_linspace(tb_float a, tb_float b, uint64_t n){
    NDShape* shape = nda_newShape(1, n);
    NDArray* x = nda_alloc(shape);

    double c;
    size_t i;

    /* step size */
    c = (b - a)/(n - 1);
    
    /* fill vector */
    for(i = 0; i < n - 1; ++i)
        x->data[i] = a + i*c;
    
    /* fix last entry to b */
    x->data[n - 1] = b;
    
    /* done */
    return x;
}

NDArray* nda_copy(NDArray* x){
    NDShape* shape = nda_copyShape(x->shape);

    uint64_t len = nda_getTotalSize(x->shape);

    NDArray* x_cpy = nda_alloc(shape);
    memcpy(x_cpy->data, x->data, len*sizeof(tb_float));

    return x_cpy;
}

NDArray* nda_ones(NDShape* shape){
    return nda_fill(shape, 1.0f);
}

struct NDArray* nda_fill(struct NDShape* shape, tb_float value){
    uint64_t len = nda_getTotalSize(shape);
    tb_float* raw = calloc(len, sizeof(tb_float));
    
    NDArray* x = calloc(1, sizeof(NDArray));
    
    size_t i = 0;
    
    for(; i < len; i++)
        raw[i] = value;
    
    x->shape = shape;
    x->data = raw;
    
    return x;
}

void nda_reshape(NDArray* x, NDShape* shape){
    NDShape* old_shape = x->shape;
    uint64_t old_len = x->shape->raw_len;
    uint64_t new_len = shape->raw_len;

    assert(old_len == new_len);

    free(old_shape->dims);
    free(old_shape);

    x->shape = shape;
}

void nda_free(NDArray* array){
    free(array->data);
    free(array->shape->dims);
    free(array->shape->strides);
    free(array->shape);
}
