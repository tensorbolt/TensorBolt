



#include <tb_session.h>
#include <tb_graph.h>

/* * * * * * * *
 * Session API *
 * * * * * * * */

// predeclaration of local functions
static TBResultNode* _run_BinaryOperation(TBGraphSession* session, TBGraph* graph, TBNode* node, TBNode* parent);
static TBResultNode* _run_UnaryOperation(TBGraphSession* session, TBGraph* graph, TBNode* node, TBNode* parent);
static TBResultNode* _run_AxisBoundOperation(TBGraphSession* session, TBGraph* graph, TBNode* node, TBNode* parent);
static TBResultNode* _run_Node(TBGraphSession* session, TBGraph* graph, TBNode* node);

TBGraphSession* tb_createLocalCPUSession(){
    TBGraphSession* session = calloc(1, sizeof(TBGraphSession));
    session->device = "/local/cpu:0";

    return session;
}

// TODO
TBResultNode* tb_runSession(TBGraphSession* session, TBGraph* graph, TBGraphNodeParam** params){
    ASSERT(graph != NULL, "Cannot run session on a NULL Graph");
    ASSERT(graph->root != NULL, "Root node of the graph %s must be NULL", graph->name);
    TBNode* root = graph->root;
    
    
    
    
    return NULL;
}

// TODO
void tb_freeSession(TBGraphSession* session){

}


static TBResultNode* _run_Node(TBGraphSession* session, TBGraph* graph, TBNode* node){
    TBNodeType type = node->type;
    
    switch(type){
        case TBNT_VARIABLE:
            
            break;
        case TBNT_CONSTANT:
            
            break;
        case TBNT_GRAPH:
            
            break;
        case TBNT_BINARY_OPERATION:
            
            break;
        case TBNT_UNARY_OPERATION:
            
            break;
        case TBNT_AXIS_BOUND_OPERATION:
            
            break;
    }
    
    return NULL;
}
