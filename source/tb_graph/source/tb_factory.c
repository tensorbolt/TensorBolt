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
 * @file tb_factory.c
 * @author Soulaymen Chouri
 * @date March 16 2019
 * @brief Factory pattern implementation of node & graph construction
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <ndarray.h>
#include <ndarray_std.h>

#include <tb_graph.h>
#include <tb_factory.h>

#define TB_ALLOC_NODE(node, t, grad, ptrVal)\
TBNode* node = calloc(1, sizeof(TBNode));\
node->calc_grad = grad;\
node->type = t;\
node->nodePtr = ptrVal;\
node->diff = NULL;

TBNode* tb_newVarNode(char* name){
	TBVariable* var = calloc(1, sizeof(TBVariable));
	var->name = name;
	
	TB_ALLOC_NODE(node, TBNT_VARIABLE, 1, var);
	
	return node;
}

TBNode* tb_newConstantNode(NDArray* array){
    ASSERT(array != NULL, "NULL array passed to create a constant node");
    TBConstant* c = calloc(1, sizeof(TBConstant));
    c->value = array;
    
    TB_ALLOC_NODE(node, TBNT_CONSTANT, 1, c);
    
    return node;
}


TBNode* tb_copyConstantNode(TBNode* con){
    ASSERT(con != NULL, "NULL node passed to copy a constant node");
    ASSERT(con->type == TBNT_CONSTANT, "Non-constant node node passed to copy a constant node");
    TBConstant* conn = (TBConstant*)con->nodePtr;
    TBConstant* c = calloc(1, sizeof(TBConstant));
    c->value = nda_copy(conn->value);
    
    TB_ALLOC_NODE(node, TBNT_CONSTANT, 1, c);
    
    return node;
}

TBNode* tb_newGraphNode(TBGraph* graph, TBGraphNodeParam** params){
    TBGraphNode * graphNode = calloc(1, sizeof(TBGraphNode));
    graphNode->graph = graph;
    graphNode->params = params;
    
	TB_ALLOC_NODE(node, TBNT_GRAPH, 1, graphNode);
	
    if(params != NULL){
        TBGraphNodeParam* param = params[0];
        uint64_t i = 0;
        while((param->node != NULL) && (param->var_name != NULL)){
            tb_graphSetVar(graph, param->node, param->var_name);
            param = params[++i];
        }
    }
    
	return node;
}

TBNode* tb_newBinaryOpNode(TBBinaryOperationType type, TBNode* lhs, TBNode* rhs){
	TBBinaryOperation* bop = calloc(1, sizeof(TBBinaryOperation));
	bop->lhs = lhs;
	bop->rhs = rhs;
	bop->type = type;
	
	TB_ALLOC_NODE(node, TBNT_BINARY_OPERATION, 1, bop);
	
	return node;
}

TBNode* tb_newUnaryOpNode(TBUnaryOperationType type, TBNode* uhs){
	TBUnaryOperation* uop = calloc(1, sizeof(TBUnaryOperation));
	uop->uhs = uhs;
	uop->type = type;
	
	TB_ALLOC_NODE(node, TBNT_UNARY_OPERATION, 1, uop);
	
	return node;
}

TBNode* tb_newAxisBoundOpNode(TBAxisBoundOperationType type, TBNode* uhs, uint64_t axis){
	TBAxisBoundOperation* abop = calloc(1, sizeof(TBAxisBoundOperation));
	abop->axis = axis;
	abop->type = type;
	abop->uhs = uhs;
	
	TB_ALLOC_NODE(node, TBNT_AXIS_BOUND_OPERATION, 1, abop);
	
	return node;
}

TBNode* tb_newTransposeOpNode(TBNode* uhs, uint64_t axis1, uint64_t axis2){
    TBTransposeOperation* top = calloc(1, sizeof(TBTransposeOperation));
    top->axis1 = axis1;
    top->axis2 = axis2;
    top->uhs = uhs;
    
    TB_ALLOC_NODE(node, TBNT_AXES_TRANSPOSE, 1, top);
    
    return node;
}

TBResultNode* tb_newResultNode(NDArray* array){
    TBResultNode* res = calloc(1, sizeof(TBResultNode));
    res->error = NULL;
    res->value = array;
    
    return res;
}

// TODO: free node impl
TBResultNode* tb_newErrorResultNode(TBErrorType errType, const char* msg, TBNode* node, TBGraph* graph){
    TBResultNode* res = calloc(1, sizeof(TBResultNode));
    res->error = calloc(1, sizeof(TBError));
    res->error->errorType = errType;
    res->error->faultyNode = node;
    res->error->graph = graph;
    res->error->message = strdup(msg);
    
    res->value = NULL;
    
    return res;
}

#undef TB_ALLOC_NODE
