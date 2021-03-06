
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
 * @file tb_ops_cpu.c
 * @author Soulaymen Chouri
 * @date March 21 2019
 * @brief File containing concrete implementation of graph operations on the CPU.
 */


#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include <cblas.h>

#include <ndarray.h>
#include <ndarray_std.h>

#include <tb_session.h>
#include <tb_graph.h>
#include <tb_operation.h>
#include <tb_ops.h>
#include <tb_factory.h>

#if tb_float == float
#define POW powf
#else
#define POW pow
#endif

/* * * * * *
 * HELPERS *
 * * * * * */

static void inline _td_map_array(tb_float* src, tb_float* dest, uint64_t len, tb_float(*f)(tb_float)){
    uint64_t i = 0;
    for(;i<len;i++)
        dest[i] = f(src[i]);
}

static inline tb_float _log(tb_float x){
#if tb_float == float
    return logf(x);
#else
    return log(x);
#endif
}

static inline tb_float _exp(tb_float x){
#if tb_float == float
    return expf(x);
#else
    return exp(x);
#endif
}

static inline tb_float _cos(tb_float x){
#if tb_float == float
    return cosf(x);
#else
    return cos(x);
#endif
}

static inline tb_float _sin(tb_float x){
#if tb_float == float
    return sinf(x);
#else
    return sin(x);
#endif
}

static inline tb_float _tan(tb_float x){
#if tb_float == float
    return tanf(x);
#else
    return tan(x);
#endif
}

static inline tb_float _tanh(tb_float x){
#if tb_float == float
    return tanhf(x);
#else
    return tanh(x);
#endif
}

static inline tb_float _relu(tb_float x){
    return (x>0)*x;
}

static inline tb_float _sigmoid(tb_float x){
    return 1.0f/(1.0f+_exp(-x));
}

static inline tb_float _dxrelu(tb_float x){
    return (x > 0)?1.0f:0.0f;
}

static inline tb_float _softplus(tb_float x){
    return _log(1.0f + _exp(x));
}

// TODO:
static inline tb_float _elu(tb_float x){
    return 0;
}

static inline tb_float _negative(tb_float x){
    return -x;
}

static NDArray* _tb_prepareBroadcast(uint8_t res, NDArray* lhs, NDArray* rhs, NDArray* biggerArray, NDArray* smallerArray){
    NDShape* lhsShape = lhs->shape;
    NDShape* rhsShape = rhs->shape;
    
    NDShape* biggerShape  = res==1?lhsShape:rhsShape;
    NDShape* smallerShape = res==1?rhsShape:lhsShape;
    
    biggerArray  = res==1?lhs:rhs;
    smallerArray = res==1?rhs:lhs;
    
    uint64_t vshape_len = biggerShape->rank;
    uint64_t num_pads = biggerShape->rank-smallerShape->rank;
    
    uint64_t* array = calloc(vshape_len, sizeof(uint64_t));
    
    size_t i = 0;
    size_t j = 0;
    
    for(; i < num_pads; i++){
        array[i] = biggerShape->dims[i];
    }
    
    for(; i < vshape_len; i++,j++){
        //printf("%lld, %lld\n", biggerShape->dims[i], smallerShape->dims[j]);
        if(biggerShape->dims[i] > smallerShape->dims[j]){
            array[i] = biggerShape->dims[i];
        }
        else {
            array[i] = smallerShape->dims[j];
        }
    }
    
    NDShape* vshape = nda_newShapeFromArray(vshape_len, array);
    //nda_debugShape(vshape);
    
    // creating output arrray
    return nda_alloc(vshape);
    
}

/* * * * * * * * * * *
 * BINARY OPERATIONS *
 * * * * * * * * * * */

TBResultNode* _tb_add(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs){
    NDShape* lhsShape = lhs->value->shape;
    NDShape* rhsShape = rhs->value->shape;
    
    uint8_t res = nda_shapeCanBroadCast(lhsShape, rhsShape);
    
    if(!res){
        char msg[1024] = {0};
        char* lhsShapeInfo = nda_shapeToString(lhsShape);
        char* rhsShapeInfo = nda_shapeToString(rhsShape);
        snprintf(msg, 1024, "Cannot broadcast shapes %s and %s", lhsShapeInfo, rhsShapeInfo);
        
        free(lhsShapeInfo);
        free(rhsShapeInfo);
        
        return tb_newErrorResultNode(TBET_INCOMPATIBLE_DIMENTIONS_EXCEPTION, msg, node, graph);
    }
    NDArray* biggerArray = NULL, * smallerArray = NULL;
    NDArray* arr_res = NULL;
    
    arr_res = _tb_prepareBroadcast(res, lhs->value, rhs->value, biggerArray, smallerArray);
    
    ASSERT(arr_res != NULL, "Fatal Error Occured.");
    
    NDShape* vshape = arr_res->shape;

    tb_float* arr = arr_res->data;
    
    size_t counter = 0;
    
    for(; counter<vshape->raw_len;counter++){
        size_t i = vshape->rank;
        uint64_t mul = vshape->raw_len;
        uint64_t counter_tmp = counter;
        uint64_t* index = calloc(vshape->rank, sizeof(uint64_t));
        for(; i !=0 ; i--){
            mul /= vshape->dims[vshape->rank - i];
            index[vshape->rank - i] = counter_tmp / mul;
            counter_tmp -= index[vshape->rank - i] * mul;
        }
        free(index);
        
        arr[counter] = nda_vget(lhs->value, index, vshape) + nda_vget(rhs->value, index, vshape);
    }
    
    return tb_newResultNode(arr_res);
}


TBResultNode* _tb_sub(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs){
    NDShape* lhsShape = lhs->value->shape;
    NDShape* rhsShape = rhs->value->shape;
    
    uint8_t res = nda_shapeCanBroadCast(lhsShape, rhsShape);
    
    if(!res){
        char msg[1024] = {0};
        char* lhsShapeInfo = nda_shapeToString(lhsShape);
        char* rhsShapeInfo = nda_shapeToString(rhsShape);
        snprintf(msg, 1024, "Cannot broadcast shapes %s and %s", lhsShapeInfo, rhsShapeInfo);
        
        free(lhsShapeInfo);
        free(rhsShapeInfo);
        
        return tb_newErrorResultNode(TBET_INCOMPATIBLE_DIMENTIONS_EXCEPTION, msg, node, graph);
    }
    NDArray* biggerArray = NULL, * smallerArray = NULL;
    NDArray* arr_res = NULL;
    
    arr_res = _tb_prepareBroadcast(res, lhs->value, rhs->value, biggerArray, smallerArray);
    
    ASSERT(arr_res != NULL, "Fatal Error Occured.");
    
    NDShape* vshape = arr_res->shape;
    
    tb_float* arr = arr_res->data;
    
    size_t counter = 0;
    
    for(; counter<vshape->raw_len;counter++){
        size_t i = vshape->rank;
        uint64_t mul = vshape->raw_len;
        uint64_t counter_tmp = counter;
        uint64_t* index = calloc(vshape->rank, sizeof(uint64_t));
        for(; i !=0 ; i--){
            mul /= vshape->dims[vshape->rank - i];
            index[vshape->rank - i] = counter_tmp / mul;
            counter_tmp -= index[vshape->rank - i] * mul;
        }
        free(index);
        
        arr[counter] = nda_vget(lhs->value, index, vshape) - nda_vget(rhs->value, index, vshape);
    }
    
    return tb_newResultNode(arr_res);
}

TBResultNode* _tb_div(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs){
    NDShape* lhsShape = lhs->value->shape;
    NDShape* rhsShape = rhs->value->shape;
    
    uint8_t res = nda_shapeCanBroadCast(lhsShape, rhsShape);
    
    if(!res){
        char msg[1024] = {0};
        char* lhsShapeInfo = nda_shapeToString(lhsShape);
        char* rhsShapeInfo = nda_shapeToString(rhsShape);
        snprintf(msg, 1024, "Cannot broadcast shapes %s and %s", lhsShapeInfo, rhsShapeInfo);
        
        free(lhsShapeInfo);
        free(rhsShapeInfo);
        
        return tb_newErrorResultNode(TBET_INCOMPATIBLE_DIMENTIONS_EXCEPTION, msg, node, graph);
    }
    NDArray* biggerArray = NULL, * smallerArray = NULL;
    NDArray* arr_res = NULL;
    
    arr_res = _tb_prepareBroadcast(res, lhs->value, rhs->value, biggerArray, smallerArray);
    
    ASSERT(arr_res != NULL, "Fatal Error Occured.");
    
    NDShape* vshape = arr_res->shape;
    
    tb_float* arr = arr_res->data;
    
    size_t counter = 0;
    
    for(; counter<vshape->raw_len;counter++){
        size_t i = vshape->rank;
        uint64_t mul = vshape->raw_len;
        uint64_t counter_tmp = counter;
        uint64_t* index = calloc(vshape->rank, sizeof(uint64_t));
        for(; i !=0 ; i--){
            mul /= vshape->dims[vshape->rank - i];
            index[vshape->rank - i] = counter_tmp / mul;
            counter_tmp -= index[vshape->rank - i] * mul;
        }
        free(index);
        
        arr[counter] = nda_vget(lhs->value, index, vshape) / nda_vget(rhs->value, index, vshape);
    }
    
    return tb_newResultNode(arr_res);
    
    return NULL;
}

TBResultNode* _tb_dot(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs){
    NDShape* lhsShape = lhs->value->shape;
    NDShape* rhsShape = rhs->value->shape;
    
    ASSERT((lhsShape->rank <= 2) && (rhsShape->rank <= 2), "Cannot perform DOT product on shapes of ranks (%"PRIu64", %"PRIu64")", lhsShape->rank, rhsShape->rank);
    
    uint64_t lhsRows = 1;
    uint64_t lhsCols = lhsShape->dims[0];
    
    if(lhsShape->rank > 1){
        lhsRows = lhsShape->dims[0];
        lhsCols = lhsShape->dims[1];
    }
    
    
    uint64_t rhsRows = 1;
    uint64_t rhsCols = rhsShape->dims[0];
    
    if(rhsShape->rank > 1){
        rhsRows = rhsShape->dims[0];
        rhsCols = rhsShape->dims[1];
    }
    
    
    ASSERT((lhsCols == rhsRows), "Cannot perform DOT product on shapes (%"PRIu64", %"PRIu64") .  (%"PRIu64", %"PRIu64")", lhsRows, lhsCols, rhsRows, rhsCols);
    
    NDArray* res_arr = NULL;
    
    // when calculating the dot product over a vector as LHS, the output is a vector by default, unless LHS has been
    // reshaped into a matrix of 1,m
    
    if(lhsShape->rank == 1){
        res_arr = nda_alloc(nda_newShape(1, rhsCols));
    }
    else{
        res_arr = nda_alloc(nda_newShape(2, lhsRows, rhsCols));
    }
    
    
#if TB_TYPE == TB_FLOAT
#define GEMM cblas_sgemm
#else
#define GEMM cblas_dgemm
#endif
    
    GEMM(CblasRowMajor,
         CblasNoTrans, CblasNoTrans, lhsRows, rhsCols, lhsCols,
         1.0, lhs->value->data, lhsCols, rhs->value->data, rhsCols, 0.0, res_arr->data, rhsCols);

    
#undef GEMM
    return tb_newResultNode(res_arr);
}

TBResultNode* _tb_mul(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs){
    NDShape* lhsShape = lhs->value->shape;
    NDShape* rhsShape = rhs->value->shape;
    
    uint8_t res = nda_shapeCanBroadCast(lhsShape, rhsShape);
    
    if(!res){
        char msg[1024] = {0};
        char* lhsShapeInfo = nda_shapeToString(lhsShape);
        char* rhsShapeInfo = nda_shapeToString(rhsShape);
        snprintf(msg, 1024, "Cannot broadcast shapes %s and %s", lhsShapeInfo, rhsShapeInfo);
        
        free(lhsShapeInfo);
        free(rhsShapeInfo);
        
        return tb_newErrorResultNode(TBET_INCOMPATIBLE_DIMENTIONS_EXCEPTION, msg, node, graph);
    }
    NDArray* biggerArray = NULL, * smallerArray = NULL;
    NDArray* arr_res = NULL;
    
    arr_res = _tb_prepareBroadcast(res, lhs->value, rhs->value, biggerArray, smallerArray);
    
    ASSERT(arr_res != NULL, "Fatal Error Occured.");
    
    NDShape* vshape = arr_res->shape;
    
    tb_float* arr = arr_res->data;
    
    size_t counter = 0;
    
    for(; counter<vshape->raw_len;counter++){
        size_t i = vshape->rank;
        uint64_t mul = vshape->raw_len;
        uint64_t counter_tmp = counter;
        uint64_t* index = calloc(vshape->rank, sizeof(uint64_t));
        for(; i !=0 ; i--){
            mul /= vshape->dims[vshape->rank - i];
            index[vshape->rank - i] = counter_tmp / mul;
            counter_tmp -= index[vshape->rank - i] * mul;
        }
        free(index);
        
        arr[counter] = nda_vget(lhs->value, index, vshape) * nda_vget(rhs->value, index, vshape);
    }
    
    return tb_newResultNode(arr_res);
    
    return NULL;
}

TBResultNode* _tb_pow(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs){
    NDShape* lhsShape = lhs->value->shape;
    NDShape* rhsShape = rhs->value->shape;
    
    uint8_t res = nda_shapeCanBroadCast(lhsShape, rhsShape);
    
    if(!res){
        char msg[1024] = {0};
        char* lhsShapeInfo = nda_shapeToString(lhsShape);
        char* rhsShapeInfo = nda_shapeToString(rhsShape);
        snprintf(msg, 1024, "Cannot broadcast shapes %s and %s", lhsShapeInfo, rhsShapeInfo);
        
        free(lhsShapeInfo);
        free(rhsShapeInfo);
        
        return tb_newErrorResultNode(TBET_INCOMPATIBLE_DIMENTIONS_EXCEPTION, msg, node, graph);
    }
    NDArray* biggerArray = NULL, * smallerArray = NULL;
    NDArray* arr_res = NULL;
    
    arr_res = _tb_prepareBroadcast(res, lhs->value, rhs->value, biggerArray, smallerArray);
    
    ASSERT(arr_res != NULL, "Fatal Error Occured.");
    
    NDShape* vshape = arr_res->shape;
    
    tb_float* arr = arr_res->data;
    
    size_t counter = 0;
    
    for(; counter<vshape->raw_len;counter++){
        size_t i = vshape->rank;
        uint64_t mul = vshape->raw_len;
        uint64_t counter_tmp = counter;
        uint64_t* index = calloc(vshape->rank, sizeof(uint64_t));
        for(; i !=0 ; i--){
            mul /= vshape->dims[vshape->rank - i];
            index[vshape->rank - i] = counter_tmp / mul;
            counter_tmp -= index[vshape->rank - i] * mul;
        }
        free(index);
        
        arr[counter] = POW(nda_vget(lhs->value, index, vshape), nda_vget(rhs->value, index, vshape));
    }
    
    return tb_newResultNode(arr_res);
    
    return NULL;
}

/* * * * * * * * * * * * * *
 * AXIS-BOUNDED OPERATIONS *
 * * * * * * * * * * * * * */

TBResultNode* _tb_max(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs, TBAxisBoundOperation* abop){
    NDShape* shape = uhs->value->shape;
    NDArray* arr = uhs->value;
    
    uint64_t axis = abop->axis;
    
    ASSERT(axis < shape->rank, "Cannot compute axis bound operation MAX on axis %lld >= array of rank %llu", axis, shape->rank);
    
    uint64_t* new_dims = NULL;
    uint64_t new_rank = 1;
    if(shape->rank == 1){
        new_dims = calloc(1, sizeof(uint64_t));
        new_dims[0] = 1;
    }
    else{
        new_dims = calloc(shape->rank-1, sizeof(uint64_t));
        new_rank =shape->rank-1;
    }
    
    uint64_t i = 0;
    uint64_t j = 0;
    for(; i < shape->rank; i++){
        if (i != axis){
            new_dims[j++] = shape->dims[i];
        }
    }
    
    NDShape* newShape = nda_newShapeFromArray(new_rank, new_dims);
    NDArray* new_arr = nda_alloc(newShape);
    
    i = 0;
    uint64_t k = 0;
    for(; i < newShape->raw_len; i++){
        
        uint64_t m = shape->rank;
        uint64_t mod = i;
        j = 0; //i/shape->strides[axis] % shape->dims[axis];
        for(;m > 0; m--){
            if(m-1 == axis){
                continue;
            }
            else{
                j += (mod%shape->dims[m-1])*shape->strides[m-1];
                mod = mod/shape->dims[m-1];
            }
        }
        new_arr->data[i] = arr->data[j];
    }
    
    k = 0;
    i = 0;
    
    for(; i < newShape->raw_len; i++){
        
        uint64_t m = shape->rank;
        uint64_t mod = i;
        j = 0; //i/shape->strides[axis] % shape->dims[axis];
        for(;m > 0; m--){
            if(m-1 == axis){
                continue;
            }
            else{
                j += (mod%shape->dims[m-1])*shape->strides[m-1];
                mod = mod/shape->dims[m-1];
                //printf("%lld, %lld, m= %lld, %f\n", i, j, m, arr->data[j]);
            }
        }
        
        j+=shape->strides[axis];
        for(k=1; k < shape->dims[axis]; j+=shape->strides[axis],k++){
            //printf("%lld, %lld, %f\n", i, j,  arr->data[j]);
            new_arr->data[i] = (new_arr->data[i]>arr->data[j])?new_arr->data[i]:arr->data[j];
        }
    }
    
    
    return tb_newResultNode(new_arr);
}

TBResultNode* _tb_min(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs, TBAxisBoundOperation* abop){
    NDShape* shape = uhs->value->shape;
    NDArray* arr = uhs->value;
    
    uint64_t axis = abop->axis;
    
    ASSERT(axis < shape->rank, "Cannot compute axis bound operation MAX on axis %lld >= array of rank %llu", axis, shape->rank);
    
    uint64_t* new_dims = NULL;
    uint64_t new_rank = 1;
    if(shape->rank == 1){
        new_dims = calloc(1, sizeof(uint64_t));
        new_dims[0] = 1;
    }
    else{
        new_dims = calloc(shape->rank-1, sizeof(uint64_t));
        new_rank =shape->rank-1;
    }
    
    uint64_t i = 0;
    uint64_t j = 0;
    for(; i < shape->rank; i++){
        if (i != axis){
            new_dims[j++] = shape->dims[i];
        }
    }
    
    NDShape* newShape = nda_newShapeFromArray(new_rank, new_dims);
    NDArray* new_arr = nda_alloc(newShape);
    
    i = 0;
    uint64_t k = 0;
    for(; i < newShape->raw_len; i++){
        
        uint64_t m = shape->rank;
        uint64_t mod = i;
        j = 0; //i/shape->strides[axis] % shape->dims[axis];
        for(;m > 0; m--){
            if(m-1 == axis){
                continue;
            }
            else{
                j += (mod%shape->dims[m-1])*shape->strides[m-1];
                mod = mod/shape->dims[m-1];
            }
        }
        new_arr->data[i] = arr->data[j];
    }
    
    k = 0;
    i = 0;
    
    for(; i < newShape->raw_len; i++){
        
        uint64_t m = shape->rank;
        uint64_t mod = i;
        j = 0; //i/shape->strides[axis] % shape->dims[axis];
        for(;m > 0; m--){
            if(m-1 == axis){
                continue;
            }
            else{
                j += (mod%shape->dims[m-1])*shape->strides[m-1];
                mod = mod/shape->dims[m-1];
                //printf("%lld, %lld, m= %lld, %f\n", i, j, m, arr->data[j]);
            }
        }
        
        j+=shape->strides[axis];
        for(k=1; k < shape->dims[axis]; j+=shape->strides[axis],k++){
            //printf("%lld, %lld, %f\n", i, j,  arr->data[j]);
            new_arr->data[i] = (new_arr->data[i]<arr->data[j])?new_arr->data[i]:arr->data[j];
        }
    }
    
    return tb_newResultNode(new_arr);
}

TBResultNode* _tb_sum(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs, TBAxisBoundOperation* abop){
    NDShape* shape = uhs->value->shape;
    NDArray* arr = uhs->value;
    
    uint64_t axis = abop->axis;
    
    ASSERT(axis < shape->rank, "Cannot compute axis bound operation SUM on axis %lld >= array of rank %llu", axis, shape->rank);
    
    uint64_t* new_dims = NULL;
    uint64_t new_rank = 1;
    if(shape->rank == 1){
        new_dims = calloc(1, sizeof(uint64_t));
        new_dims[0] = 1;
    }
    else{
        new_dims = calloc(shape->rank-1, sizeof(uint64_t));
        new_rank =shape->rank-1;
    }
    
    uint64_t i = 0;
    uint64_t j = 0;
    for(; i < shape->rank; i++){
        if (i != axis){
            new_dims[j++] = shape->dims[i];
        }
    }
    
    NDShape* newShape = nda_newShapeFromArray(new_rank, new_dims);
    NDArray* new_arr = nda_alloc(newShape);
    
    i = 0;
    uint64_t k = 0;
    for(; i < newShape->raw_len; i++){
        
        uint64_t m = shape->rank;
        uint64_t mod = i;
        j = 0; //i/shape->strides[axis] % shape->dims[axis];
        for(;m > 0; m--){
            if(m-1 == axis){
                continue;
            }
            else{
                j += (mod%shape->dims[m-1])*shape->strides[m-1];
                mod = mod/shape->dims[m-1];
                //printf("%lld, %lld, m= %lld, %f\n", i, j, m, arr->data[j]);
            }
        }
        
        
        for(k=0; k < shape->dims[axis]; j+=shape->strides[axis],k++){
            //printf("%lld, %lld, %f\n", i, j,  arr->data[j]);
            new_arr->data[i] += arr->data[j];
        }
    }
    
    return tb_newResultNode(new_arr);
}

TBResultNode* _tb_mean(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs, TBAxisBoundOperation* abop){
    return NULL;
}

TBResultNode* _tb_argmax(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs, TBAxisBoundOperation* abop){
    NDShape* shape = uhs->value->shape;
    //NDArray* arr = uhs->value;
    
    uint64_t axis = abop->axis;
    
    ASSERT(axis < shape->rank, "Cannot compute axis bound operation MAX on axis %lld >= array of rank %llu", axis, shape->rank);
    
    uint64_t* new_dims = NULL;
    uint64_t new_rank = 1;
    if(shape->rank == 1){
        new_dims = calloc(1, sizeof(uint64_t));
        new_dims[0] = 1;
    }
    else{
        new_dims = calloc(shape->rank-1, sizeof(uint64_t));
        new_rank =shape->rank-1;
    }
    
    uint64_t i = 0;
    uint64_t j = 0;
    for(; i < shape->rank; i++){
        if (i != axis){
            new_dims[j++] = shape->dims[i];
        }
    }
    
    NDShape* newShape = nda_newShapeFromArray(new_rank, new_dims);
    NDArray* new_arr = nda_alloc(newShape);
    
    i = 0;
    uint64_t k = 0;
    for(; i < newShape->raw_len; i++){
        
        uint64_t m = shape->rank;
        uint64_t mod = i;
        j = 0; //i/shape->strides[axis] % shape->dims[axis];
        for(;m > 0; m--){
            if(m-1 == axis){
                continue;
            }
            else{
                j += (mod%shape->dims[m-1])*shape->strides[m-1];
                mod = mod/shape->dims[m-1];
            }
        }
        new_arr->data[i] = 0;
    }
    
    k = 0;
    i = 0;
    
    for(; i < newShape->raw_len; i++){
        
        uint64_t m = shape->rank;
        uint64_t mod = i;
        j = 0; //i/shape->strides[axis] % shape->dims[axis];
        for(;m > 0; m--){
            if(m-1 == axis){
                continue;
            }
            else{
                j += (mod%shape->dims[m-1])*shape->strides[m-1];
                mod = mod/shape->dims[m-1];
                //printf("%lld, %lld, m= %lld, %f\n", i, j, m, arr->data[j]);
            }
        }
        
        j+=shape->strides[axis];
        for(k=1; k < shape->dims[axis]; j+=shape->strides[axis],k++){
            //printf("%lld, %lld, %f\n", i, j,  arr->data[j]);
            new_arr->data[i] = (new_arr->data[i]>k)?new_arr->data[i]:k;
        }
    }
    
    return tb_newResultNode(new_arr);
}


TBResultNode* _tb_argmin(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs, TBAxisBoundOperation* abop){
    NDShape* shape = uhs->value->shape;
    //NDArray* arr = uhs->value;
    
    uint64_t axis = abop->axis;
    
    ASSERT(axis < shape->rank, "Cannot compute axis bound operation MAX on axis %lld >= array of rank %llu", axis, shape->rank);
    
    uint64_t* new_dims = NULL;
    uint64_t new_rank = 1;
    if(shape->rank == 1){
        new_dims = calloc(1, sizeof(uint64_t));
        new_dims[0] = 1;
    }
    else{
        new_dims = calloc(shape->rank-1, sizeof(uint64_t));
        new_rank =shape->rank-1;
    }
    
    uint64_t i = 0;
    uint64_t j = 0;
    for(; i < shape->rank; i++){
        if (i != axis){
            new_dims[j++] = shape->dims[i];
        }
    }
    
    NDShape* newShape = nda_newShapeFromArray(new_rank, new_dims);
    NDArray* new_arr = nda_alloc(newShape);
    
    i = 0;
    uint64_t k = 0;
    for(; i < newShape->raw_len; i++){
        
        uint64_t m = shape->rank;
        uint64_t mod = i;
        j = 0; //i/shape->strides[axis] % shape->dims[axis];
        for(;m > 0; m--){
            if(m-1 == axis){
                continue;
            }
            else{
                j += (mod%shape->dims[m-1])*shape->strides[m-1];
                mod = mod/shape->dims[m-1];
            }
        }
        new_arr->data[i] = 0;
    }
    
    k = 0;
    i = 0;
    
    for(; i < newShape->raw_len; i++){
        
        uint64_t m = shape->rank;
        uint64_t mod = i;
        j = 0; //i/shape->strides[axis] % shape->dims[axis];
        for(;m > 0; m--){
            if(m-1 == axis){
                continue;
            }
            else{
                j += (mod%shape->dims[m-1])*shape->strides[m-1];
                mod = mod/shape->dims[m-1];
                //printf("%lld, %lld, m= %lld, %f\n", i, j, m, arr->data[j]);
            }
        }
        
        j+=shape->strides[axis];
        for(k=1; k < shape->dims[axis]; j+=shape->strides[axis],k++){
            //printf("%lld, %lld, %f\n", i, j,  arr->data[j]);
            new_arr->data[i] = (new_arr->data[i]<k)?new_arr->data[i]:k;
        }
    }
    
    return tb_newResultNode(new_arr);
}

TBResultNode* _tb_product(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs, TBAxisBoundOperation* abop){
    NDShape* shape = uhs->value->shape;
    NDArray* arr = uhs->value;
    
    uint64_t axis = abop->axis;
    
    ASSERT(axis < shape->rank, "Cannot compute axis bound operation SUM on axis %lld >= array of rank %llu", axis, shape->rank);
    
    uint64_t* new_dims = NULL;
    uint64_t new_rank = 1;
    if(shape->rank == 1){
        new_dims = calloc(1, sizeof(uint64_t));
        new_dims[0] = 1;
    }
    else{
        new_dims = calloc(shape->rank-1, sizeof(uint64_t));
        new_rank =shape->rank-1;
    }
    
    uint64_t i = 0;
    uint64_t j = 0;
    for(; i < shape->rank; i++){
        if (i != axis){
            new_dims[j++] = shape->dims[i];
        }
    }
    
    NDShape* newShape = nda_newShapeFromArray(new_rank, new_dims);
    NDArray* new_arr = nda_alloc(newShape);
    
    i = 0;
    uint64_t k = 0;
    for(; i < newShape->raw_len; i++){
        new_arr->data[i] = 1;
    }
    
    for(i=0; i < newShape->raw_len; i++){
        
        uint64_t m = shape->rank;
        uint64_t mod = i;
        j = 0; //i/shape->strides[axis] % shape->dims[axis];
        for(;m > 0; m--){
            if(m-1 == axis){
                continue;
            }
            else{
                j += (mod%shape->dims[m-1])*shape->strides[m-1];
                mod = mod/shape->dims[m-1];
                //printf("%lld, %lld, m= %lld, %f\n", i, j, m, arr->data[j]);
            }
        }
        
        
        for(k=0; k < shape->dims[axis]; j+=shape->strides[axis],k++){
            //printf("%lld, %lld, %f\n", i, j,  arr->data[j]);
            new_arr->data[i] *= arr->data[j];
        }
    }
    
    return tb_newResultNode(new_arr);
}

TBResultNode* _tb_softmax(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs, TBAxisBoundOperation* abop){
    return NULL;
}

/* * * * * * * * * * *
 * UNARY  OPERATIONS *
 * * * * * * * * * * */

#define TB_UNARY_OP_MAP(func_name, elt_func)\
TBResultNode* func_name(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs){\
    NDArray* x = nda_copy(uhs->value);\
    _td_map_array(uhs->value->data, x->data, x->shape->raw_len, elt_func);\
\
    TBResultNode* res = tb_newResultNode(x);\
\
    return res;\
}

TB_UNARY_OP_MAP(_tb_negative, _negative);
TB_UNARY_OP_MAP(_tb_sin, _sin);
TB_UNARY_OP_MAP(_tb_cos, _cos);
TB_UNARY_OP_MAP(_tb_exp, _exp);
TB_UNARY_OP_MAP(_tb_log, _log);
TB_UNARY_OP_MAP(_tb_tan, _tan);
TB_UNARY_OP_MAP(_tb_tanh, _tanh);
TB_UNARY_OP_MAP(_tb_relu, _relu);
TB_UNARY_OP_MAP(_tb_softplus, _softplus);
TB_UNARY_OP_MAP(_tb_sigmoid, _sigmoid);
TB_UNARY_OP_MAP(_tb_dxrelu, _dxrelu);

TBResultNode* _tb_transpose(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs, TBTransposeOperation* top){
    NDArray* arr = uhs->value;
    NDArray* arr_res = nda_copy(arr);
    NDShape* shape = arr_res->shape;
    if(shape->rank == 1){
        NDShape* newShape = nda_newShape(2, 1, shape->dims[0]);
        nda_reshape(arr_res, newShape);
        shape = newShape;
        // TODO: free shape
        //return tb_newResultNode(arr_res);
    }
    
    shape = arr_res->shape;
    
    uint64_t idim = shape->dims[top->axis1];
    shape->dims[top->axis1] = shape->dims[top->axis2];
    shape->dims[top->axis2] = idim;
    
    idim = shape->strides[top->axis1];
    shape->strides[top->axis1] = shape->strides[top->axis2];
    shape->strides[top->axis2] = idim;
    
    return tb_newResultNode(arr_res);
}

TBResultNode* _tb_elu(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs){
    return NULL;
}


#undef TB_UNARY_OP_MAP
