#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>

#include <tb_graph.h>
#include <map.h>
#include <vec.h>

void tb_assert(int cond, const char * rawcond, const char * fmt, ...){
    if(cond)
		return;
	char temp[1024];
    va_list vl;
    va_start(vl, fmt);
    vsprintf(temp, fmt, vl);
    va_end(vl);
	fprintf(stdout, "Fatal error, assertion failed: %s\n", rawcond);
    fprintf(stdout, temp);
    fprintf(stdout, "\n");
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