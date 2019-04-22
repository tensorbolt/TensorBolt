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
node->nodePtr = ptrVal;

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

/*
 * TODO: Implement TBGraphNodeParam binding
 */
TBNode* tb_newGraphNode(TBGraph* graph, TBGraphNodeParam** params){
	ASSERT(params == NULL, "second argument should be NULL as it is not supported yet");
    TBGraphNode * graphNode = calloc(1, sizeof(TBGraphNode));
    graphNode->graph = graph;
    graphNode->params = params;
    
	TB_ALLOC_NODE(node, TBNT_GRAPH, 1, graphNode);
	
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

TBNode* tb_newAxisBoundNode(TBAxisBoundOperationType type, TBNode* uhs, uint64_t axis){
	TBAxisBoundOperation* abop = calloc(1, sizeof(TBAxisBoundOperation));
	abop->axis = axis;
	abop->type = type;
	abop->uhs = uhs;
	
	TB_ALLOC_NODE(node, TBNT_AXIS_BOUND_OPERATION, 1, abop);
	
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
