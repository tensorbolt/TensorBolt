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
 * @file tb_session_cpu.c
 * @author Soulaymen Chouri
 * @date March 16 2019
 * @brief File containing Sessions metadata & implementations.
 */


#include <string.h>
#include <stdint.h>
#include <stdio.h>

#include <tb_session.h>
#include <tb_graph.h>
#include <tb_factory.h>
#include <tb_ops.h>

#include <ndarray.h>
#include <ndarray_std.h>

/* * * * * * * *
 * Session API *
 * * * * * * * */

// predeclaration of local functions
static TBResultNode* _run_BinaryOperation(TBGraphSession* session, TBGraph* graph, TBNode* node);
static TBResultNode* _run_UnaryOperation(TBGraphSession* session, TBGraph* graph, TBNode* node);
static TBResultNode* _run_AxisBoundOperation(TBGraphSession* session, TBGraph* graph, TBNode* node);
static TBResultNode* _run_TransposeOperation(TBGraphSession* session, TBGraph* graph, TBNode* node);
static TBResultNode* _run_Node(TBGraphSession* session, TBGraph* graph, TBNode* node);

TBGraphSession* tb_createLocalCPUSession(){
    TBGraphSession* session = calloc(1, sizeof(TBGraphSession));

    return session;
}

// TODO
// free graph & nodes
TBResultNode* tb_runSession(TBGraphSession* session, TBGraph* graph, TBGraphNodeParam** params){
    ASSERT(graph != NULL, "Cannot run session on a NULL Graph");
    ASSERT(graph->root != NULL, "Root node of the graph %s must be NULL", graph->name);
    
    if(params != NULL){
        size_t i = 0;
        
        for(;(params[i]->node != NULL) && (params[i]->var_name != NULL);i++){
            tb_graphSetVar(graph, params[i]->node, params[i]->var_name);
        }
    }
    
    TBNode* root = graph->root;
    
    tb_storeNodesInGraph(graph, root);
    
    // TODO: free stuff
    
    return _run_Node(session, graph, root);
}

struct TBResultNode* tb_runSessionNodeOnly(struct TBGraphSession* session, struct TBNode* node){
    TBGraph* g_tmp = tb_newGraph("tmp0001", node);
    TBResultNode* res = tb_runSession(session, g_tmp, NULL);
    
    // TODO: free stuff
    
    return res;
}


/* * * * * * * * * * * * *
 * Graph Processing  API *
 * * * * * * * * * * * * */

static TBResultNode* _run_Node(TBGraphSession* session, TBGraph* graph, TBNode* node){
    TBNodeType type = node->type;
    TBResultNode* res = NULL;
    
    switch(type){
        case TBNT_VARIABLE:{
            TBVariable* var = (TBVariable*)node->nodePtr;
            TBNode* n = tb_graphGetVar(graph, var->name);
            
            if(n == NULL){
                char msg[1024] = {0};
                snprintf(msg, 1024, "Graph `%s` runtime error, variable `%s` does not exist", graph->name, var->name);
                return tb_newErrorResultNode(TBET_VARIABLE_DOES_NOT_EXIST, msg, node, graph);
            }
            
            res = _run_Node(session, graph, n);
            
            break;
        }
        case TBNT_CONSTANT:
            res = tb_newResultNode(((TBConstant*)node->nodePtr)->value);
            break;
        case TBNT_GRAPH:
        {
            TBGraphNode * graphNode = (TBGraphNode*)node->nodePtr;
            TBGraph* g = graphNode->graph;
            
            ASSERT(g != NULL, "Cannot start NULL nested graph");
            
            res = tb_runSession(session, g, graphNode->params);
            break;
        }
        case TBNT_BINARY_OPERATION:
            res =  _run_BinaryOperation(session, graph, node);
            break;
        case TBNT_UNARY_OPERATION:
            res =  _run_UnaryOperation(session, graph, node);
            break;
        case TBNT_AXIS_BOUND_OPERATION:
            res =  _run_AxisBoundOperation(session, graph, node);
            break;
        case TBNT_AXES_TRANSPOSE:
            res =  _run_TransposeOperation(session, graph, node);
            break;
    }
    
    if(node->diff == NULL){
        node->diff = tb_newResultNode(nda_alloc(res->value->shape));
    }
    
    node->result = tb_newResultNode(nda_copy(res->value));
    
    return res;
}

static TBResultNode* _run_UnaryOperation(TBGraphSession* session, TBGraph* graph, TBNode* node){
    TBUnaryOperation* op = (TBUnaryOperation*)node->nodePtr;
    TBResultNode* uhs = _run_Node(session, graph, op->uhs);
    
    switch(op->type){
        case TBUOT_MINUS:
            return _tb_negative(session, graph, node, uhs);
        case TBUOT_EXP:
            return _tb_exp(session, graph, node, uhs);
        case TBUOT_LOG:
            return _tb_log(session, graph, node, uhs);
        case TBUOT_SIN:
            return _tb_sin(session, graph, node, uhs);
        case TBUOT_COS:
            return _tb_cos(session, graph, node, uhs);
        case TBUOT_TAN:
            return _tb_tan(session, graph, node, uhs);
        case TBUOT_TANH:
            return _tb_tanh(session, graph, node, uhs);
        case TBUOT_RELU:
            return _tb_relu(session, graph, node, uhs);
        case TBUOT_SOFTPLUS:
            return _tb_softplus(session, graph, node, uhs);
        case TBUOT_SIGMOID:
            return _tb_sigmoid(session, graph, node, uhs);
        case TBUOT_DXRELU:
            return _tb_dxrelu(session, graph, node, uhs);
    }
}

static TBResultNode* _run_BinaryOperation(TBGraphSession* session, TBGraph* graph, TBNode* node){
    TBBinaryOperation* op = (TBBinaryOperation*)node->nodePtr;
    TBResultNode* lhs = _run_Node(session, graph, op->lhs);
    TBResultNode* rhs = _run_Node(session, graph, op->rhs);
    
    switch(op->type){
        case TBBOT_ADD:
            return _tb_add(session, graph, node, lhs, rhs);
            break;
        case TBBOT_SUB:
            return _tb_sub(session, graph, node, lhs, rhs);
            break;
        case TBBOT_MULT:
            return _tb_mul(session, graph, node, lhs, rhs);
            break;
        case TBBOT_DIV:
            return _tb_div(session, graph, node, lhs, rhs);
            break;
        case TBBOT_POW:
            return _tb_pow(session, graph, node, lhs, rhs);
            break;
        case TBBOT_DOT:
            return _tb_dot(session, graph, node, lhs, rhs);
            break;
    }
    return NULL;
}

static TBResultNode* _run_AxisBoundOperation(TBGraphSession* session, TBGraph* graph, TBNode* node){
    TBAxisBoundOperation* abop = (TBAxisBoundOperation*)node->nodePtr;
    TBResultNode* uhs = _run_Node(session, graph, abop->uhs);
    
    switch(abop->type){
        case TBABOT_SUM:
            return _tb_sum(session, graph, node, uhs, abop);
            break;
        case TBABOT_PRODUCT:
            return _tb_product(session, graph, node, uhs, abop);
            break;
        case TBABOT_MIN:
            return _tb_min(session, graph, node, uhs, abop);
            break;
        case TBABOT_MAX:
            return _tb_max(session, graph, node, uhs, abop);
            break;
        case TBABOT_MEAN:
            return _tb_mean(session, graph, node, uhs, abop);
            break;
        case TBABOT_VARIANCE:
            // TODO: Implement
            return NULL;
            break;
        case TBABOT_SOFTMAX:
            return _tb_softmax(session, graph, node, uhs, abop);
            break;
        case TBABOT_ARGMIN:
            return _tb_argmin(session, graph, node, uhs, abop);
            break;
        case TBABOT_ARGMAX:
            return _tb_argmax(session, graph, node, uhs, abop);
            break;
    }
    return NULL;
}

static TBResultNode* _run_TransposeOperation(TBGraphSession* session, TBGraph* graph, TBNode* node){
    TBTransposeOperation* top = (TBTransposeOperation*)node->nodePtr;
    TBResultNode* uhs = _run_Node(session, graph, top->uhs);
    
    return _tb_transpose(session, graph, node, uhs, top);
}

void tb_freeSession(struct TBGraphSession* session){
    free(session);
}
