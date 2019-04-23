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
 * @file tb_factory.h
 * @author Soulaymen Chouri
 * @date March 20 2019
 * @brief File for creating nodes and graphs, factory pattern.
 * This file provides methods for creating tensors, objects and nodes.
 */

#ifndef _TB_FACTORY_H_
#define _TB_FACTORY_H_

#include <ndarray.h>
#include <tb_graph.h>
#include <tb_operation.h>

/**
 * \brief Creates a new variable
 * \param[in] name variable name
 * \return new variable node
 */
TBNode* tb_newVarNode(char* name);

/**
 * \brief Creates a new constant value
 * \param[in] array ND Array
 * \return new variable node
 */
TBNode* tb_newConstantNode(struct NDArray* array);

/**
 * \brief Creates a new graph node
 * \param[in] graph Nested graph to use
 * \param[in] params node-variable name pairs to be assigned from the parent graph to the child graph.
 * \return new graph node
 */
TBNode* tb_newGraphNode(TBGraph* graph, TBGraphNodeParam** params);

/**
 * \brief Creates a binary operation node
 * \param[in] type binary operation type
 * \param[in] lhs left hand side operand
 * \param[in] rhs right hand side operand
 * \return new binary operation node
 */
TBNode* tb_newBinaryOpNode(TBBinaryOperationType type, TBNode* lhs, TBNode* rhs);

/**
 * \brief Creates an unary operation node 
 * \param[in] type unary operation type
 * \param[in] uhs unary hand side operand
 * \return new unary operation node
 */
TBNode* tb_newUnaryOpNode(TBUnaryOperationType type, TBNode* uhs);

/**
 * \brief Creates an Axis-bounded operation which operates on a given dimension/axis
 * \param[in] type operation type
 * \param[in] uhs unary hand side
 * \param[in] axis Axis or dimension on which to operate
 * \return new Axis bound node
 */
TBNode* tb_newAxisBoundOpNode(TBAxisBoundOperationType type, TBNode* uhs, uint64_t axis);

/**
 * \brief Creates an operation node which transposes an NDArray
 * \param[in] uhs Unary hand side node
 * \param[in] axis1, must be lower than the rank of the array
 * \param[in] axis2, must be lower than the rank of the array
 * \return new Transpose node
 */
TBNode* tb_newTransposeOpNode(TBNode* uhs, uint64_t axis1, uint64_t axis2);

/**
 * \brief Creates a new result node (do not create in case of error)
 * \param[in] array NDArray, value of the result
 */
TBResultNode* tb_newResultNode(struct NDArray* array);

/**
 * \brief Create a runtime error result node, which does NOT stop the execution! i.e Not an assertion
 * \param[in] errType type of the exception/error
 * \param[in] msg Error message description
 * \param[in] node Which node caused the error
 * \param[in] graph In which graph the exception occured
 * \return Error result node
 */
TBResultNode* tb_newErrorResultNode(TBErrorType errType, const char* msg, TBNode* node, TBGraph* graph);

#endif
