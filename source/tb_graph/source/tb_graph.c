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

    // TODO: add nodes to the list
    vec_init(&graph->nodes);

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
	}
}
