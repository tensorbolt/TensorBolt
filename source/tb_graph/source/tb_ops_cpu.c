
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

#include <tb_session.h>
#include <tb_graph.h>
#include <tb_operation.h>
#include <tb_ops.h>
#include <tb_factory.h>

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


/* * * * * * * * * * *
 * BINARY OPERATIONS *
 * * * * * * * * * * */

TBResultNode* _tb_add(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs){
    NDShape* lhsShape = lhs->value->shape;
    NDShape* rhsShape = rhs->value->shape;
    
    if((lhsShape->rank == 1) && (rhsShape->dims[0] == 1)){
        if (lhsShape->dims[0] == lhsShape->dims[0]){
            return 1;
        }
        else {
            return 0;
        }
    }
    
    return NULL;
}


TBResultNode* _tb_sub(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs){
    NDShape* lhsShape = lhs->value->shape;
    NDShape* rhsShape = rhs->value->shape;
    
    return NULL;
}

TBResultNode* _tb_div(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs){
    NDShape* lhsShape = lhs->value->shape;
    NDShape* rhsShape = rhs->value->shape;
    
    return NULL;
}

TBResultNode* _tb_dot(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs){
    NDShape* lhsShape = lhs->value->shape;
    NDShape* rhsShape = rhs->value->shape;
    
    return NULL;
}

TBResultNode* _tb_mul(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs){
    NDShape* lhsShape = lhs->value->shape;
    NDShape* rhsShape = rhs->value->shape;
    
    return NULL;
}

TBResultNode* _tb_pow(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs){
    NDShape* lhsShape = lhs->value->shape;
    NDShape* rhsShape = rhs->value->shape;
    
    return NULL;
}

/* * * * * * * * * * * * * *
 * AXIS-BOUNDED OPERATIONS *
 * * * * * * * * * * * * * */

TBResultNode* _tb_max(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs){
    return NULL;
}

TBResultNode* _tb_min(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs){
    return NULL;
}

TBResultNode* _tb_sum(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs){
    return NULL;
}


TBResultNode* _tb_mean(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs){
    return NULL;
}


TBResultNode* _tb_argmax(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs){
    return NULL;
}


TBResultNode* _tb_argmin(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs){
    return NULL;
}


/* * * * * * * * * * *
 * UNARY  OPERATIONS *
 * * * * * * * * * * */

#define TB_UNARY_OP_MAP(func_name, elt_func)\
TBResultNode* func_name(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs){\
    NDArray* x = nda_copy(uhs->value);\
    _td_map_array(uhs->value->data, x->data, nda_getTotalSize(x->shape), elt_func);\
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

TBResultNode* _tb_transpose(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs){
    return NULL;
}

TBResultNode* _tb_elu(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs){
    return NULL;
}


#undef TB_UNARY_OP_MAP
