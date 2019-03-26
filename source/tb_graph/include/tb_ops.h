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
 * @file tb_ops
 * @author Soulaymen Chouri
 * @date March 21 2019
 * @brief File containing concrete implementation of graph operations on the CPU.
 */

#ifndef _TB_OPS_H_
#define _TB_OPS_H_

#include <stdint.h>

#include <tb_session.h>
#include <tb_graph.h>
#include <tb_operation.h>

/* * * * * * * * * * *
 * BINARY OPERATIONS *
 * * * * * * * * * * */

/**
 * \brief Adds two result nodes i.e NDArrays. Supports broadcasting.
 * \param[in] sess Session which contains the context of execution
 * \param[in] graph Parent graph which is being executed
 * \param[in] node Current node that is being executed
 * \param[in] lhs Left-hand side result node
 * \param[in] rhs Right-hand side result node
 * \return Result of the operation
 */
TBResultNode* _tb_add(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs);


/**
 * \brief Substracts two result nodes i.e NDArrays. Supports broadcasting.
 * \param[in] sess Session which contains the context of execution
 * \param[in] graph Parent graph which is being executed
 * \param[in] node Current node that is being executed
 * \param[in] lhs Left-hand side result node
 * \param[in] rhs Right-hand side result node
 * \return Result of the operation
 */
TBResultNode* _tb_sub(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs);

/**
 * \brief Divides two result nodes i.e NDArrays. Can also detects division by zero and alert is such cases. Supports broadcasting.
 * \param[in] sess Session which contains the context of execution
 * \param[in] graph Parent graph which is being executed
 * \param[in] node Current node that is being executed
 * \param[in] lhs Left-hand side result node
 * \param[in] rhs Right-hand side result node
 * \return Result of the operation
 */
TBResultNode* _tb_div(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs);

/**
 * \brief Dot product between two tensors. Support limited broadcasting, shapes must matches the mathematical rules.
 * \param[in] sess Session which contains the context of execution
 * \param[in] graph Parent graph which is being executed
 * \param[in] node Current node that is being executed
 * \param[in] lhs Left-hand side result node
 * \param[in] rhs Right-hand side result node
 * \return Result of the operation
 */
TBResultNode* _tb_dot(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs);

/**
 * \brief Multiplication between two tensors. Support broadcasting
 * \param[in] sess Session which contains the context of execution
 * \param[in] graph Parent graph which is being executed
 * \param[in] node Current node that is being executed
 * \param[in] lhs Left-hand side result node
 * \param[in] rhs Right-hand side result node
 * \return Result of the operation
 */
TBResultNode* _tb_mul(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs);

/**
 * \brief Power between two tensors. Support broadcasting, evern if RHS is not a scalar
 * \param[in] sess Session which contains the context of execution
 * \param[in] graph Parent graph which is being executed
 * \param[in] node Current node that is being executed
 * \param[in] lhs Left-hand side result node
 * \param[in] rhs Right-hand side result node
 * \return Result of the operation
 */
TBResultNode* _tb_pow(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* lhs, TBResultNode* rhs);

/* * * * * * * * * * * * * *
 * AXIS-BOUNDED OPERATIONS *
 * * * * * * * * * * * * * */

/**
 * \brief Maximum elements through the given axis
 * \param[in] sess Session which contains the context of execution
 * \param[in] graph Parent graph which is being executed
 * \param[in] node Current node that is being executed
 * \param[in] uhs Unary-hand side result node
 * \return Result of the operation
 */
TBResultNode* _tb_max(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);

/**
 * \brief Minimum element in the given axis
 * \param[in] sess Session which contains the context of execution
 * \param[in] graph Parent graph which is being executed
 * \param[in] node Current node that is being executed
 * \param[in] uhs Unary-hand side result node
 * \return Result of the operation
 */
TBResultNode* _tb_min(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);

/**
 * \brief Sum of the element in the given axis
 * \param[in] sess Session which contains the context of execution
 * \param[in] graph Parent graph which is being executed
 * \param[in] node Current node that is being executed
 * \param[in] uhs Unary-hand side result node
 * \return Result of the operation
 */
TBResultNode* _tb_sum(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);

/**
 * \brief Mean of the element in the given axis
 * \param[in] sess Session which contains the context of execution
 * \param[in] graph Parent graph which is being executed
 * \param[in] node Current node that is being executed
 * \param[in] uhs Unary-hand side result node
 * \return Result of the operation
 */
TBResultNode* _tb_mean(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);

/**
 * \brief Argmax element in the given axis
 * \param[in] sess Session which contains the context of execution
 * \param[in] graph Parent graph which is being executed
 * \param[in] node Current node that is being executed
 * \param[in] uhs Unary-hand side result node
 * \return Result of the operation
 */
TBResultNode* _tb_argmax(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);

/**
 * \brief Argmin element in the given axis
 * \param[in] sess Session which contains the context of execution
 * \param[in] graph Parent graph which is being executed
 * \param[in] node Current node that is being executed
 * \param[in] uhs Unary-hand side result node
 * \return Result of the operation
 */
TBResultNode* _tb_argmin(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);


/* * * * * * * * * * *
 * UNARY  OPERATIONS *
 * * * * * * * * * * */

TBResultNode* _tb_negative(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);
TBResultNode* _tb_transpose(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);
TBResultNode* _tb_sin(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);
TBResultNode* _tb_cos(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);
TBResultNode* _tb_exp(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);
TBResultNode* _tb_log(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);
TBResultNode* _tb_tan(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);
TBResultNode* _tb_tanh(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);
TBResultNode* _tb_relu(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);
TBResultNode* _tb_softplus(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);
TBResultNode* _tb_sigmoid(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);
TBResultNode* _tb_elu(TBGraphSession* sess, TBGraph* graph, TBNode* node, TBResultNode* uhs);

#endif