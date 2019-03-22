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
 * @file tb_operation.h
 * @author Soulaymen Chouri
 * @date March 16 2019
 * @brief File containing Graph operations data structure
 *
 * This file defines every possible node operation in the graph
 */

#ifndef _TB_OPERATIONS_H_
#define _TB_OPERATIONS_H_

#include <stdint.h>

#include <ndarray.h>

/**
 * \brief List of the available binary operation types
 */
typedef enum TBBinaryOperationType {
	TBBOT_ADD=0,  /**< ADD (+broadcast) */
	TBBOT_SUB,    /**< SUBSTRACTION (+broadcast) */
	TBBOT_MULT,   /**< MULTIPLICATION (+bradcast) */
	TBBOT_DIV,    /**< DIVISION (+broadcast) */
	TBBOT_POW,    /**< Element-wise Power RHS must be a scalar. */
	TBBOT_DOT,    /**< DOT PRODUCT i.e matmul */
} TBBinaryOperationType;
#define MAX_BINARY_OPERATION TBBOT_DOT

/**
 * \brief List of the available unary operation types
 */
typedef enum TBUnaryOperationType {
	TBUOT_MINUS=0,    /**< Element-wise minus */
	TBUOT_TRANSPOSE,  /**< Transpose, applies only to matrices */
	TBUOT_EXP,        /**< Element-wise Exponential */
	TBUOT_LOG,        /**< Element-wise log */
	TBUOT_SIN,        /**< Element-wise sin */
	TBUOT_COS,        /**< Element-wise cos */
	TBUOT_TAN,        /**< Element-wise tan */
	TBUOT_TANH,       /**< Element-wise tanh */
	TBUOT_RELU,       /**< Element-wise RELU */
	TBUOT_SOFTPLUS,   /**< Element-wise softplus */
	TBUOT_SIGMOID,    /**< Element-wise sigmoid */
} TBUnaryOperationType;

#define MAX_UNARY_OPERATION TBUOT_SOFTPLUS

/**
 * \brief List of operations that apply mainly to a given axis of tensor
 */
typedef enum TBAxisBoundOperationType {
	TBABOT_SUM = 0,   /**< SUM */
	TBABOT_PRODUCT,   /**< PRODUCT */
	TBABOT_MIN,       /**< MIN */
	TBABOT_MAX,       /**< MAX */
	TBABOT_MEAN,      /**< MEAN */
	TBABOT_VARIANCE,  /**< VARIANCE*/
	TBABOT_SOFTMAX,   /**< Numerically stable SOFTMAX */
	TBABOT_ARGMIN,    /**< ARGMIN */
	TBABOT_ARGMAX,    /**< ARGMAX */
}TBAxisBoundOperationType;

#define MAX_AXIS_BOUND_OPERATION TBABOT_VARIANCE


/**
 * \brief Binary operation node
 */
typedef struct TBBinaryOperation {
	struct TBNode* lhs;               /**< LHS */
	struct TBNode* rhs;               /**< RHS */
	
	TBBinaryOperationType type;       /**< Type of the operation */
}TBBinaryOperation;

typedef struct TBUnaryOperation {
	struct TBNode* uhs;               /**< UHS */
	TBUnaryOperationType type;        /**< Type of the operation */
}TBUnaryOperation;

typedef struct TBAxisBoundOperation{
	struct TBNode* uhs;               /**< UHS */
	uint64_t axis;                     /**< Axis on whichto perform the operation */
	
	TBAxisBoundOperationType type;    /**< Type of the operation */
}TBAxisBoundOperation;

/**
 * \brief variable node
 */
typedef struct TBVariable {
	char* name; /**< Name of the variable */
} TBVariable;

/**
 * \brief constant node
 */
typedef struct TBConstant {
	struct NDArray* value;  /**< Constant node value */
}TBConstant;

/**
 * \brief Graph Node
 */
typedef struct TBGraphNode{
    TBGraphNodeParam** params;   /**< Graph arguments, nodes should be constants!!!!!! */
    TBGraph* graph;              /**< Actual graph structure */
}TBGraphNode;

/**
 * \brief Specifies if CPU information where correctly extracted or not
 */
typedef enum CPUInfoAvailabilty {
    TBCPUINFO_AVAILABLE,
    TBCPUINFO_NOT_AVAILABLE
}CPUInfoAvailabilty;

/**
 * \brief Specifies various CPU features that can be extracted.
 */
typedef enum CPUFeatures {
    TBCPUF_MMX = 0,
    TBCPUF_MMX_EXT,
    TBCPUF_SSE,
    TBCPUF_SSE2,
    TBCPUF_SSE3,
    TBCPUF_3DNOW,
    TBCPUF_AVX,
    TBCPUF_SSSE3,
    TBCPUF_SSE4_1,
    TBCPUF_SSE4_2,
}CPUFeatures;


#endif
