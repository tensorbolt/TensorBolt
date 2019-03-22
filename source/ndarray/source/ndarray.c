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

#include "ndarray.h"

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

    return shape;
}

void nda_debugShape(NDShape* shape){
    printf("Tensor{ .rank = %llu, ", shape->rank);
    
    size_t i = 0;
    
    for(; i < shape->rank; i++){
        printf(".dims[%zu] = %llu, ", i, shape->dims[i]);
    }

    printf("}\n");
}


void nda_debugValue(NDArray* tensor){
    nda_debugShape(tensor->shape);
    uint64_t len = nda_getTotalSize(tensor->shape);

    size_t i = 0;

    for(; i < len; i++){
        printf("\t[%zu] = %f,\n", i, tensor->data[i]);
    }

    printf("\n");
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
    
    return shape2;
}

NDArray* nda_alloc(NDShape* shape){
    uint64_t len = nda_getTotalSize(shape);
    tb_float* raw = calloc(len, sizeof(tb_float));

    NDArray* x = calloc(1, sizeof(NDArray));

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


void nda_reshape(NDArray* x, NDShape* shape){
    NDShape* old_shape = x->shape;
    uint64_t old_len = nda_getTotalSize(x->shape);
    uint64_t new_len = nda_getTotalSize(shape);

    assert(old_len == new_len);

    free(old_shape->dims);
    free(old_shape);

    x->shape = shape;
}

void nda_free(NDArray* array){
    free(array->data);
    free(array->shape->dims);
    free(array->shape);
    free(array);
}
