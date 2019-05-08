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
 * @file tb_autograd.c
 * @author Soulaymen Chouri
 * @date May 1st 2019
 * @brief File containing Automatic differenciation
 */

#include <ndarray.h>
#include <ndarray_std.h>

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include <tb_autograd.h>
#include <tb_graph.h>
#include <tb_factory.h>
#include <tb_operation.h>


static void _tb_freeNodeDiff(TBNode* node){
    tb_freeResultNode(NULL, node->diff);
    free(node->diff);
}

static TBNode* _tb_adaptDiffToShape(TBNode* diff_node, NDShape* diffShape, NDShape* valueShape){
    int64_t d = (int64_t)valueShape->rank - (int64_t)diffShape->rank;
    if (d < 0)
        d = -d;
    
    uint64_t k = (int64_t)d;
    
    while(d > 0){
        diff_node =tb_newAxisBoundOpNode(TBABOT_SUM, diff_node, 0);
        d--;
    }
    
    uint64_t i = 0;
    for(; i < diffShape->rank; i++){
        if(diffShape->dims[i] < valueShape->dims[i+k]){
            diff_node = tb_newAxisBoundOpNode(TBABOT_SUM, diff_node, i);
        }
    }
    
    return diff_node;
}

static TBNode* _tb_convertResultNodeToNode(TBResultNode* res){
    return tb_newConstantNode(nda_copy(res->value));
}

static TBResultNode* _tb_convertNodetoResultNode(TBNode* res){
    ASSERT(res->type == TBNT_CONSTANT, "Cannot convert non constant node to result node");
    return tb_newResultNode(nda_copy(((TBConstant*)res->nodePtr)->value));
}

static void _tb_autograd_bop_add(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBBinaryOperation* bop){
    NDShape* lhsDiffShape = bop->lhs->diff->value->shape;
    NDShape* rhsDiffShape = bop->rhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(bop->lhs->diff),
                                       _tb_convertResultNodeToNode(node->diff)
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, lhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    _tb_freeNodeDiff(bop->lhs);
    nda_reshape(res1->value, nda_copyShape(bop->lhs->result->value->shape));
    bop->lhs->diff = (res1);
    
    TBNode* mult2 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(bop->rhs->diff),
                                       _tb_convertResultNodeToNode(node->diff)
                                       );
    
    mult2 = _tb_adaptDiffToShape(mult2, rhsDiffShape, node->result->value->shape);
    
    TBResultNode* res2 = tb_runSessionNodeOnly(session, mult2);
    _tb_freeNodeDiff(bop->rhs);
    nda_reshape(res2->value, nda_copyShape(bop->rhs->result->value->shape));
    bop->rhs->diff = (res2);
}


static void _tb_autograd_bop_sub(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBBinaryOperation* bop){
    NDShape* lhsDiffShape = bop->lhs->diff->value->shape;
    NDShape* rhsDiffShape = bop->rhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(bop->lhs->diff),
                                       _tb_convertResultNodeToNode(node->diff)
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, lhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    _tb_freeNodeDiff(bop->lhs);
    nda_reshape(res1->value, nda_copyShape(bop->lhs->result->value->shape));
    bop->lhs->diff = (res1);
    
    TBNode* mult2 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(bop->rhs->diff),
                                       tb_newUnaryOpNode(TBUOT_MINUS,
                                                         _tb_convertResultNodeToNode(node->diff)
                                                         )
                                       );
    
    mult2 = _tb_adaptDiffToShape(mult2, rhsDiffShape, node->result->value->shape);
    
    TBResultNode* res2 = tb_runSessionNodeOnly(session, mult2);
    nda_reshape(res2->value, nda_copyShape(bop->rhs->result->value->shape));
    _tb_freeNodeDiff(bop->rhs);
    bop->rhs->diff = (res2);
}

static void _tb_autograd_bop_mult(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBBinaryOperation* bop){
    NDShape* lhsDiffShape = bop->lhs->diff->value->shape;
    NDShape* rhsDiffShape = bop->rhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(bop->lhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          _tb_convertResultNodeToNode(bop->rhs->result)
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, lhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(bop->lhs->result->value->shape));
    _tb_freeNodeDiff(bop->lhs);
    bop->lhs->diff = (res1);
    
    TBNode* mult2 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(bop->rhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          _tb_convertResultNodeToNode(bop->lhs->result)
                                                          )
                                       );
    
    mult2 = _tb_adaptDiffToShape(mult2, rhsDiffShape, node->result->value->shape);
    
    TBResultNode* res2 = tb_runSessionNodeOnly(session, mult2);
    nda_reshape(res2->value, nda_copyShape(bop->rhs->result->value->shape));
    _tb_freeNodeDiff(bop->rhs);
    bop->rhs->diff = (res2);
}

static void _tb_autograd_bop_div(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBBinaryOperation* bop){
    NDShape* lhsDiffShape = bop->lhs->diff->value->shape;
    NDShape* rhsDiffShape = bop->rhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(bop->lhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newBinaryOpNode(TBBOT_DIV,
                                                                             tb_newConstantNode(nda_ones(nda_newShape(1, 1))),
                                                                             _tb_convertResultNodeToNode(bop->rhs->result)
                                                                             )
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, lhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(bop->lhs->result->value->shape));
    _tb_freeNodeDiff(bop->lhs);
    bop->lhs->diff = (res1);
    
    TBNode* mult2 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(bop->rhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          tb_newUnaryOpNode(TBUOT_MINUS,
                                                                            tb_newBinaryOpNode(TBBOT_DIV,
                                                                                               _tb_convertResultNodeToNode(bop->lhs->result),
                                                                                               tb_newBinaryOpNode(TBBOT_POW,
                                                                                                                  _tb_convertResultNodeToNode(bop->rhs->result),
                                                                                                                  tb_newConstantNode(nda_fill(nda_newShape(1, 1), 2))
                                                                                                                  )
                                                                                               )
                                                                            ),
                                                          _tb_convertResultNodeToNode(node->diff)
                                                          )
                                       );
    
    mult2 = _tb_adaptDiffToShape(mult2, rhsDiffShape, node->result->value->shape);
    
    TBResultNode* res2 = tb_runSessionNodeOnly(session, mult2);
    nda_reshape(res2->value, nda_copyShape(bop->rhs->result->value->shape));
    _tb_freeNodeDiff(bop->rhs);
    bop->rhs->diff = (res2);
}


static void _tb_autograd_bop_pow(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBBinaryOperation* bop){
    NDShape* lhsDiffShape = bop->lhs->diff->value->shape;
    NDShape* rhsDiffShape = bop->rhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(bop->lhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newBinaryOpNode(TBBOT_MULT,
                                                                             _tb_convertResultNodeToNode(bop->rhs->result),
                                                                             tb_newBinaryOpNode(TBBOT_POW,
                                                                                                _tb_convertResultNodeToNode(bop->lhs->result),
                                                                                                tb_newBinaryOpNode(TBBOT_SUB,
                                                                                                                   _tb_convertResultNodeToNode(bop->rhs->result),
                                                                                                                   tb_newConstantNode(nda_ones(nda_newShape(1, 1)))
                                                                                                                   )
                                                                                                )
                                                                             )
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, lhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(bop->lhs->result->value->shape));
    _tb_freeNodeDiff(bop->lhs);
    bop->lhs->diff = (res1);
    
    TBNode* mult2 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(bop->rhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newBinaryOpNode(TBBOT_MULT,
                                                                             _tb_convertResultNodeToNode(node->result),
                                                                             tb_newUnaryOpNode(TBUOT_LOG,
                                                                                               _tb_convertResultNodeToNode(bop->lhs->result)
                                                                                               )
                                                                             )
                                                          )
                                       );
    
    mult2 = _tb_adaptDiffToShape(mult2, rhsDiffShape, node->result->value->shape);
    
    TBResultNode* res2 = tb_runSessionNodeOnly(session, mult2);
    nda_reshape(res2->value, nda_copyShape(bop->rhs->result->value->shape));
    _tb_freeNodeDiff(bop->rhs);
    bop->rhs->diff = (res2);
}

static void _tb_autograd_bop_dot(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBBinaryOperation* bop){
    NDShape* lhsDiffShape = bop->lhs->diff->value->shape;
    NDShape* rhsDiffShape = bop->rhs->diff->value->shape;
   
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(bop->lhs->diff),
                                       tb_newBinaryOpNode(TBBOT_DOT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newTransposeOpNode(_tb_convertResultNodeToNode(bop->rhs->result), 1, 0)
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, lhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(bop->lhs->result->value->shape));
    _tb_freeNodeDiff(bop->lhs);
    bop->lhs->diff = (res1);
    
    TBNode* mult2 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(bop->rhs->diff),
                                       tb_newBinaryOpNode(TBBOT_DOT,
                                                          tb_newTransposeOpNode(_tb_convertResultNodeToNode(bop->lhs->result), 1, 0),
                                                          _tb_convertResultNodeToNode(node->diff)
                                                          )
                                       );
    
    mult2 = _tb_adaptDiffToShape(mult2, rhsDiffShape, node->result->value->shape);
    
    TBResultNode* res2 = tb_runSessionNodeOnly(session, mult2);
    nda_reshape(res2->value, nda_copyShape(bop->rhs->result->value->shape));
    _tb_freeNodeDiff(bop->rhs);
    bop->rhs->diff = (res2);
}

static void _tb_autograd_uop_minus(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBUnaryOperation* uop){
    NDShape* uhsDiffShape = uop->uhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(uop->uhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newConstantNode(nda_fill(nda_newShape(1, 1), -1.0f))
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, uhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(uop->uhs->result->value->shape));
    _tb_freeNodeDiff(uop->uhs);
    uop->uhs->diff = (res1);
}

static void _tb_autograd_uop_exp(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBUnaryOperation* uop){
    NDShape* uhsDiffShape = uop->uhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(uop->uhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          _tb_convertResultNodeToNode(node->result)
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, uhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(uop->uhs->result->value->shape));
    _tb_freeNodeDiff(uop->uhs);
    uop->uhs->diff = (res1);
}

static void _tb_autograd_uop_log(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBUnaryOperation* uop){
    NDShape* uhsDiffShape = uop->uhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(uop->uhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newBinaryOpNode(TBBOT_DIV,
                                                                             tb_newConstantNode(nda_ones(nda_newShape(1, 1))),
                                                                             _tb_convertResultNodeToNode(node->result))
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, uhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(uop->uhs->result->value->shape));
    _tb_freeNodeDiff(uop->uhs);
    uop->uhs->diff = (res1);
}

static void _tb_autograd_uop_sin(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBUnaryOperation* uop){
    NDShape* uhsDiffShape = uop->uhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(uop->uhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newUnaryOpNode(TBUOT_COS,
                                                                             _tb_convertResultNodeToNode(uop->uhs->result))
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, uhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(uop->uhs->result->value->shape));
    _tb_freeNodeDiff(uop->uhs);
    uop->uhs->diff = (res1);
}

static void _tb_autograd_uop_cos(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBUnaryOperation* uop){
    NDShape* uhsDiffShape = uop->uhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(uop->uhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newUnaryOpNode(TBUOT_MINUS,
                                                                            tb_newUnaryOpNode(TBUOT_SIN,
                                                                                              _tb_convertResultNodeToNode(uop->uhs->result))
                                                                            )
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, uhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(uop->uhs->result->value->shape));
    _tb_freeNodeDiff(uop->uhs);
    uop->uhs->diff = (res1);
}

static void _tb_autograd_uop_tan(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBUnaryOperation* uop){
    NDShape* uhsDiffShape = uop->uhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(uop->uhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newBinaryOpNode(TBBOT_DIV,
                                                                             tb_newConstantNode(nda_ones(nda_newShape(1,1))),
                                                                             tb_newBinaryOpNode(TBBOT_POW,
                                                                                                tb_newUnaryOpNode(TBUOT_COS, _tb_convertResultNodeToNode(uop->uhs->result)),
                                                                                                tb_newConstantNode(nda_fill(nda_newShape(1, 1), 2.0f))
                                                                                                )
                                                                             )
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, uhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(uop->uhs->result->value->shape));
    _tb_freeNodeDiff(uop->uhs);
    uop->uhs->diff = (res1);
}

static void _tb_autograd_uop_tanh(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBUnaryOperation* uop){
    NDShape* uhsDiffShape = uop->uhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(uop->uhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newBinaryOpNode(TBBOT_SUB,
                                                                             tb_newConstantNode(nda_fill(nda_newShape(1, 1), 1.0f)),
                                                                             tb_newBinaryOpNode(TBBOT_POW,
                                                                                                _tb_convertResultNodeToNode(uop->uhs->result),
                                                                                                tb_newConstantNode(nda_fill(nda_newShape(1, 1), 2.0f))
                                                                                                )
                                                                             )
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, uhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(uop->uhs->result->value->shape));
    _tb_freeNodeDiff(uop->uhs);
    uop->uhs->diff = (res1);
}

static void _tb_autograd_uop_relu(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBUnaryOperation* uop){
    NDShape* uhsDiffShape = uop->uhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(uop->uhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newUnaryOpNode(TBUOT_DXRELU, _tb_convertResultNodeToNode(uop->uhs->result))
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, uhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(uop->uhs->result->value->shape));
    _tb_freeNodeDiff(uop->uhs);
    uop->uhs->diff = (res1);
}

static void _tb_autograd_uop_softplus(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBUnaryOperation* uop){
    NDShape* uhsDiffShape = uop->uhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(uop->uhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newUnaryOpNode(TBUOT_SIGMOID, _tb_convertResultNodeToNode(uop->uhs->result))
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, uhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(uop->uhs->result->value->shape));
    _tb_freeNodeDiff(uop->uhs);
    uop->uhs->diff = (res1);
}

static void _tb_autograd_uop_sigmoid(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBUnaryOperation* uop){
    NDShape* uhsDiffShape = uop->uhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(uop->uhs->diff),
                                       tb_newBinaryOpNode(TBBOT_MULT,
                                                          _tb_convertResultNodeToNode(node->diff),
                                                          tb_newBinaryOpNode(TBBOT_MULT,
                                                                             _tb_convertResultNodeToNode(node->result),
                                                                             tb_newBinaryOpNode(TBBOT_SUB,
                                                                                                tb_newConstantNode(nda_ones(nda_newShape(1, 1))),
                                                                                                _tb_convertResultNodeToNode(node->result)
                                                                                                )
                                                                             )
                                                          )
                                       );
    
    mult1 = _tb_adaptDiffToShape(mult1, uhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(uop->uhs->result->value->shape));
    _tb_freeNodeDiff(uop->uhs);
    uop->uhs->diff = (res1);
}

static void _tb_autograd_abop_sum(struct TBGraphSession* session, TBGraph* graph, TBNode* node, TBAxisBoundOperation* abop){
    NDShape* uhsDiffShape = abop->uhs->diff->value->shape;
    
    TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                       _tb_convertResultNodeToNode(abop->uhs->diff),
                                       //tb_newTransposeOpNode(_tb_convertResultNodeToNode(node->diff), abop->axis, 1)
                                       _tb_convertResultNodeToNode(node->diff)
                                       );
    
    //mult1 = _tb_adaptDiffToShape(mult1, uhsDiffShape, node->result->value->shape);
    TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
    nda_reshape(res1->value, nda_copyShape(abop->uhs->result->value->shape));
    _tb_freeNodeDiff(abop->uhs);
    abop->uhs->diff = (res1);
}

void tb_autogradNode(struct TBGraphSession* session, TBGraph* graph, TBNode* node){
    switch(node->type){
            
        case TBNT_CONSTANT:
            printf("constant found\n");
            break;
        
        case TBNT_VARIABLE:{
            const char* name = ((TBVariable*)node->nodePtr)->name;
            TBNode* original = tb_graphGetVar(graph, name);
            original->diff = (node->diff);
            break;
        }
            
        case TBNT_GRAPH:{
            TBGraphNode* g = (TBGraphNode*)node->nodePtr;
            tb_autogradNestedGraph(session, g, node->diff);
            break;
        }
        case TBNT_BINARY_OPERATION:
        {
            TBBinaryOperation* bop = (TBBinaryOperation*)node->nodePtr;
            switch (bop->type) {
                case TBBOT_ADD:
                    _tb_autograd_bop_add(session, graph, node, bop);
                    break;
                
                case TBBOT_SUB:
                    _tb_autograd_bop_sub(session, graph, node, bop);
                    break;
                
                case TBBOT_MULT:
                    _tb_autograd_bop_mult(session, graph, node, bop);
                    break;
                    
                case TBBOT_DIV:
                    _tb_autograd_bop_div(session, graph, node, bop);
                    break;
            
                case TBBOT_POW:
                    _tb_autograd_bop_pow(session, graph, node, bop);
                    break;
                case TBBOT_DOT:
                    _tb_autograd_bop_dot(session, graph, node, bop);
                    break;
            }
            
            tb_autogradNode(session, graph, bop->lhs);
            tb_autogradNode(session, graph, bop->rhs);
            break;
        }
        case TBNT_UNARY_OPERATION:{
            TBUnaryOperation* uop = (TBUnaryOperation*)node->nodePtr;
            switch(uop->type){
                    
                case TBUOT_MINUS:
                    _tb_autograd_uop_minus(session, graph, node, uop);
                    break;
                case TBUOT_EXP:
                    _tb_autograd_uop_exp(session, graph, node, uop);
                    break;
                case TBUOT_LOG:
                    _tb_autograd_uop_log(session, graph, node, uop);
                    break;
                case TBUOT_SIN:
                    _tb_autograd_uop_sin(session, graph, node, uop);
                    break;
                case TBUOT_COS:
                    _tb_autograd_uop_cos(session, graph, node, uop);
                    break;
                case TBUOT_TAN:
                    _tb_autograd_uop_tan(session, graph, node, uop);
                    break;
                case TBUOT_TANH:
                    _tb_autograd_uop_tanh(session, graph, node, uop);
                    break;
                case TBUOT_RELU:
                    _tb_autograd_uop_relu(session, graph, node, uop);
                    break;
                case TBUOT_SOFTPLUS:
                    _tb_autograd_uop_softplus(session, graph, node, uop);
                    break;
                case TBUOT_SIGMOID:
                    _tb_autograd_uop_sigmoid(session, graph, node, uop);
                    break;
                case TBUOT_DXRELU:
                    // TODO: Throw an error
                    exit(-1);
                    break;
            }
            
            tb_autogradNode(session, graph, uop->uhs);
            break;
       }
        case TBNT_AXIS_BOUND_OPERATION:
        {
            TBAxisBoundOperation* abop = (TBAxisBoundOperation*)node->nodePtr;
            
            switch(abop->type){
                    
                case TBABOT_SUM:
                    //_tb_autograd_abop_sum(session, graph, node, abop);
                    break;
                case TBABOT_PRODUCT:
                    
                    break;
                case TBABOT_MIN:
                    
                    break;
                case TBABOT_MAX:
                    
                    break;
                case TBABOT_MEAN:
                    
                    break;
                case TBABOT_VARIANCE:
                    
                    break;
                case TBABOT_SOFTMAX:
                    
                    break;
                case TBABOT_ARGMIN:
                    
                    break;
                case TBABOT_ARGMAX:
                    
                    break;
            }
            // TODO: my brain is boiling
            exit(-1);
            tb_autogradNode(session, graph, abop->uhs);
            break;
        }
        case TBNT_AXES_TRANSPOSE:
        {
            TBTransposeOperation* top = (TBTransposeOperation*)node->nodePtr;
            NDShape* uhsDiffShape = top->uhs->diff->value->shape;
            
            TBNode* mult1 = tb_newBinaryOpNode(TBBOT_ADD,
                                               _tb_convertResultNodeToNode(top->uhs->diff),
                                               tb_newTransposeOpNode(_tb_convertResultNodeToNode(node->diff), top->axis1, top->axis2)
                                               );
            
            mult1 = _tb_adaptDiffToShape(mult1, uhsDiffShape, node->result->value->shape);
            TBResultNode* res1 = tb_runSessionNodeOnly(session, mult1);
            nda_reshape(res1->value, nda_copyShape(top->uhs->result->value->shape));
            _tb_freeNodeDiff(top->uhs);
            top->uhs->diff = (res1);
            
            tb_autogradNode(session, graph, top->uhs);
            
            break;
        }
    }
    /*printf("node type = %d\n", node->type);
    printf("node value\n");
    nda_debugValue(node->result->value);
    printf("node diff\n");
    nda_debugValue(node->diff->value);
    printf("----------\n\n");
     */
}


void tb_autogradGraph(struct TBGraphSession* session, TBGraph* graph){
    graph->root->diff = tb_newResultNode(nda_ones(nda_copyShape(graph->root->result->value->shape)));
    tb_autogradNode(session, graph, graph->root);
}

void tb_autogradNestedGraph(struct TBGraphSession* session, TBGraph* graph, TBResultNode* parentDiff){
    graph->root->diff = tb_copyResultNode(parentDiff);
    tb_autogradNode(session, graph, graph->root);
}
