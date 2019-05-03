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
 * @file tb_graph.c
 * @author Soulaymen Chouri
 * @date March 16 2019
 * @brief File containing Graph architecture implementation.
 */

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>

#include <tb_operation.h>
#include <tb_graph.h>
#include <map.h>
#include <vec.h>

void tb_assert(int cond, const char * rawcond, const char* func_name, const char * fmt, ...){
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
    
    exit(-1);
}


TBGraph* tb_newGraph(char* name, TBNode* rootNode){
    TBGraph* graph = calloc(1, sizeof(TBGraph));
    graph->name = name;
    graph->root = rootNode;

    map_init(&graph->vars);

    vec_init(&graph->nodes);

    tb_storeNodesInGraph(graph, rootNode);
    
    return graph;
}

// TODO:
void tb_freeGraph(TBGraph* graph){

}

void tb_graphSetVar(TBGraph* graph, TBNode* node, const char* name){
    TBNode** old = map_get(&graph->vars, name);
    if(old != NULL){
        //freeNode(graph, *old);
        //free(*old);
        map_remove(&graph->vars, name);
    }

    /*int res = */map_set(&graph->vars, name, node);
}

TBNode* tb_graphGetVar(TBGraph* graph, const char* name){
    TBNode** noderef = map_get(&graph->vars, name);
    if (noderef == NULL)
        return NULL;
    return *noderef;
}

void tb_storeNodesInGraph(TBGraph* graph, TBNode* node){
	int idx = -1;
	
	vec_find(&graph->nodes, node, idx);
	
	if(idx != -1)
		return;
	
	vec_push(&graph->nodes, node);
	
	
	switch(node->type){
		case TBNT_CONSTANT:
			break;
		case TBNT_VARIABLE:
			break;
		case TBNT_BINARY_OPERATION:
			tb_storeNodesInGraph(graph, ((TBBinaryOperation*)node->nodePtr)->lhs);
			tb_storeNodesInGraph(graph, ((TBBinaryOperation*)node->nodePtr)->rhs);
			break;
		case TBNT_UNARY_OPERATION:
			tb_storeNodesInGraph(graph, ((TBUnaryOperation*)node->nodePtr)->uhs);
			break;
		case TBNT_AXIS_BOUND_OPERATION:
			tb_storeNodesInGraph(graph, ((TBUnaryOperation*)node->nodePtr)->uhs);
			break;
		case TBNT_GRAPH:
			tb_storeNodesInGraph(graph, ((TBGraph*)node->nodePtr)->root);
			break;
        case TBNT_AXES_TRANSPOSE:
            tb_storeNodesInGraph(graph, ((TBTransposeOperation*)node->nodePtr)->uhs);
            break;
    }
}

void tb_freeNode(TBGraph* graph, TBNode* node){
    if (node->diff != NULL){
        tb_freeResultNode(graph, node->diff);
        free(node->diff);
    }
    
    if(node->result != NULL){
        tb_freeResultNode(graph, node->result);
        free(node->result);
    }
    
    switch(node->type){
        case TBNT_VARIABLE:
            free(node->nodePtr);
            break;
            
        case TBNT_CONSTANT:
            nda_free(((TBConstant*)node->nodePtr)->value);
            free(node->nodePtr);
            break;
            
        case TBNT_GRAPH:
            // TODO
            break;
            
        case TBNT_BINARY_OPERATION:
            free(node->nodePtr);
            break;
            
        case TBNT_UNARY_OPERATION:
            free(node->nodePtr);
            break;
            
        case TBNT_AXIS_BOUND_OPERATION:
            free(node->nodePtr);
            break;
            
        case TBNT_AXES_TRANSPOSE:
            free(node->nodePtr);
            break;
    }
}

void tb_freeResultNode(TBGraph* graph, TBResultNode* node){
    if(node->error){
        free(node->error);
    }
    
    if(node->value != NULL){
        nda_free(node->value);
        free(node->value);
    }
}

