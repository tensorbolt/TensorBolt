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
 * @file tb_graph.h
 * @author Soulaymen Chouri
 * @date March 16 2019
 * @brief File containing Graph & Node data structure.
 *
 * This file defines computational graph and nodes structures
 */

#ifndef _TB_GRAPH_H_
#define _TB_GRAPH_H_

#include <stdint.h>

#include <ndarray.h>
#include <tb_errors.h>
#include <tb_session.h>

#include "map.h"
#include "vec.h"


/* source: https://stackoverflow.com/questions/15305310/predefined-macros-for-function-name-func */
#ifndef __FUNCTION_NAME__
#ifdef WIN32   //WINDOWS
#define __FUNCTION_NAME__   __FUNCTION__
#else          //*NIX
#define __FUNCTION_NAME__   __func__
#endif
#endif

#ifdef ASSERT
#undef ASSERT
#endif

#define TB_ASSERT_LOG

#if defined(TB_ASSERT_STANDARD)
#define ASSERT(c ,msg, ...) assert(c)
#elif defined (TB_ASSERT_LOG)
#define ASSERT(c, msg, ...) tb_assert(c, #c, __FUNCTION_NAME__ , msg, ##__VA_ARGS__)
#elif defined (TB_ASSERT_NONE)
#define ASSERT(c, msg, ...)
#else
#error Please define an assertion policy.
#endif


/**
 *\brief Creates the type cgnode_vec_t for storing all nodes in a graph, for internal use.
 */
typedef vec_t(void*) TBNode_Vec;

/**
 * \brief advanced assertion strategy for debugging purposes
 * define TB_ASSERT_STANDARD to use standard C assert
 * define TB_ASSERT_LOG to use advanced logging strategy
 * define TB_ASSERT_NONE to ignore assertions
 * 
 * \param[in] cond  condition to check
 * \param[in] raw   condition string used to debug on stderr in case of error
 * \param[in] fmt   additional custom error message to be logged in stderr
 * \param[in] ...   custom error message parameters similar to printf
 */
void tb_assert(int cond, const char * rawcond, const char* func_name, const char * fmt, ...);

/* * * * * * * * * *
 * DATA STRUCTURES *
 * * * * * * * * * */ 

/**
 * \brief List of the available node types
 */
typedef enum TBNodeType{
    TBNT_VARIABLE = 0,             /**< Variable node, usually used as input. */
	TBNT_CONSTANT,                 /**< Constant tensor node. */
	TBNT_GRAPH,                    /**< Graph node, TensorBolt supports nested graphs. */
	TBNT_BINARY_OPERATION,         /**< Binary operation. */
	TBNT_UNARY_OPERATION,          /**< Unary operation. */
	TBNT_AXIS_BOUND_OPERATION,     /**< Axis-bounded operations i.e operations that are applied over a specific axis or dimension. */
    TBNT_AXES_TRANSPOSE,           /**< Transpose two axes of an NDArray */
}TBNodeType;

#define MAX_NODE_TYPE TBNT_AXES_TRANSPOSE

/**
 * \brief Node data structure
 */
typedef struct TBNode {
	TBNodeType type;               /**< Node type */
    void* nodePtr;                 /**< Pointer to the actual node structure */
    uint8_t calc_grad;             /**< Boolean flag indicating that the gradient will be calculated for this node. If it is set to false, its child will also be set to false */
    struct TBResultNode* result;   /**< Pointer to the actual result of the node, in order to improve performance . */
    struct TBResultNode* diff;     /**< Pointer to the derivative of this node w.r.t to the root node in the graph. */
}TBNode;

/**
 * \brief Result of an operation
 */
typedef struct TBResultNode {
	struct NDArray* value;         /**< Tensor value */
	TBError* error;                /**< Error pointer in case of exception during the execution */
}TBResultNode;

/**
 * \brief Graph data structure
 */
typedef struct TBGraph {
	char* name;                    /**< Graph name */
	TBNode* root;                  /**< Graph Root Node */
	map_t(TBNode*) vars;           /**< variables, a map from char* => TBNode* */
	TBNode_Vec nodes;              /**< Lookup table to free nodes later on */
}TBGraph;


/**
 * \brief pair of node and names passed to nested graphs,
 * since we do not pass the length of this array to any of
 * the functions using this structure, we check for {NULL, NULL}
 * as the end of the list.
 */
typedef struct TBGraphNodeParam {
	struct TBNode* node;           /**< Node pointer to be bounded with the next variable name */
	char* var_name;                /**< Variable name to be bounded with the previous node */
}TBGraphNodeParam;


/* * * * * * *
 * Graph API *
 * * * * * * */ 

/**
 * \brief Creates a new graph
 * \param name[in] graph name
 * \param rootNode root node of the graph
 * \return new allocated graph
 */
TBGraph* tb_newGraph(char* name, TBNode* rootNode);

/**
 * \brief Frees a graph
 * \param[in/out] graph Graph to deallocate
 */
void tb_freeGraph(TBGraph* graph);

/**
 * \brief Frees a node
 * \param [in/out] graph parent graph
 * \param[in/out] node Node to free
 */
void tb_freeNode(TBGraph* graph, TBNode* node);

/**
 * \brief Frees a result node
 * \param[in/out] node Result Node to free
 */
void tb_freeResultNode(TBGraph* graph, TBResultNode* node);

/**
 * \brief Binds a node with a variable name within the graph
 * \param[in/out] graph Graph to update
 * \param[in] node Node to bind
 * \param[in] name Variable name
 */
void tb_graphSetVar(TBGraph* graph, TBNode* node, const char* name);

/**
 * \brief Returns the value of a given variable name
 * \param[in] graph Base graph
 * \param[in] name Variable name to request
 * \return Node if it exists, NULL otherwise.
 */
TBNode* tb_graphGetVar(TBGraph* graph, const char* name);

/**
 * \brief Recursively travers a node and stores all nodes in the graph nodes list,
 * in order to make freeing them later on a piece of cake (or so I hope). This
 * function is automatically called when running a session.
 * \param[in/out] graph Graph to process
 * \param[in] node Node to add, and recursively traverse.
 */
void tb_storeNodesInGraph(TBGraph* graph, TBNode* node);

#endif
