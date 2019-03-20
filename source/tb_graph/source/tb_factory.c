#include <stdint.h>
#include <stdlib.h>

#include <tb_graph.h>
#include <ndarray.h>
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

/*
 * TODO: Implement TBGraphNodeParam binding
 */
TBNode* tb_newGraphNode(TBGraph* graph, TBGraphNodeParam** params){
	ASSERT(params == NULL, "second argument should be NULL as it is not supported yet");
	TB_ALLOC_NODE(node, TBNT_GRAPH, 1, graph);
	
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


#undef TB_ALLOC_NODE
