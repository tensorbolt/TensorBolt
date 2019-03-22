


#include <string.h>
#include <stdint.h>
#include <stdio.h>

#include <tb_session.h>
#include <tb_graph.h>
#include <tb_factory.h>
#include <tb_ops.h>

/* * * * * * * *
 * Session API *
 * * * * * * * */

// predeclaration of local functions
static TBResultNode* _run_BinaryOperation(TBGraphSession* session, TBGraph* graph, TBNode* node);
static TBResultNode* _run_UnaryOperation(TBGraphSession* session, TBGraph* graph, TBNode* node);
static TBResultNode* _run_AxisBoundOperation(TBGraphSession* session, TBGraph* graph, TBNode* node);
static TBResultNode* _run_Node(TBGraphSession* session, TBGraph* graph, TBNode* node);

TBGraphSession* tb_createLocalCPUSession(){
    TBGraphSession* session = calloc(1, sizeof(TBGraphSession));
    session->device = "/local/cpu:0";

    return session;
}

// TODO
// free graph & nodes
TBResultNode* tb_runSession(TBGraphSession* session, TBGraph* graph, TBGraphNodeParam** params){
    ASSERT(graph != NULL, "Cannot run session on a NULL Graph");
    ASSERT(graph->root != NULL, "Root node of the graph %s must be NULL", graph->name);
    TBNode* root = graph->root;
    
    tb_storeNodesInGraph(graph, root);
    
    return _run_Node(session, graph, root);
}

// TODO
void tb_freeSession(TBGraphSession* session){

}

/* * * * * * * * * * * * *
 * Graph Processing  API *
 * * * * * * * * * * * * */
 

static TBResultNode* _run_Node(TBGraphSession* session, TBGraph* graph, TBNode* node){
    TBNodeType type = node->type;
    TBResultNode* res = NULL;
    
    switch(type){
        case TBNT_VARIABLE:{
            TBVariable* var = (TBVariable*)node->nodePtr;
            TBNode* n = tb_graphGetVar(graph, var->name);
            
            if(n == NULL){
                char msg[1024] = {0};
                snprintf(msg, 1024, "Graph `%s` runtime error, variable `%s` does not exist", graph->name, var->name);
                return tb_newErrorResultNode(TBET_VARIABLE_DOES_NOT_EXIST, msg, node, graph);
            }
            
            res = _run_Node(session, graph, n);
            
            break;
        }
        case TBNT_CONSTANT:
            res = tb_newResultNode(((TBConstant*)node->nodePtr)->value);
            break;
        case TBNT_GRAPH:
            
            break;
        case TBNT_BINARY_OPERATION:
            
            break;
        case TBNT_UNARY_OPERATION:
            return _run_UnaryOperation(session, graph, node);
            break;
        case TBNT_AXIS_BOUND_OPERATION:
            
            break;
    }
    
    return res;
}

static TBResultNode* _run_UnaryOperation(TBGraphSession* session, TBGraph* graph, TBNode* node){
    TBUnaryOperation* op = (TBUnaryOperation*)node->nodePtr;
    TBResultNode* uhs = _run_Node(session, graph, op->uhs);
    
    switch(op->type){
            
        case TBUOT_MINUS:
            return _tb_negative(session, graph, node, uhs);
        case TBUOT_TRANSPOSE:
            return NULL;
            break;
        case TBUOT_EXP:
            return _tb_exp(session, graph, node, uhs);
            break;
        case TBUOT_LOG:
            return _tb_log(session, graph, node, uhs);
            break;
        case TBUOT_SIN:
            return _tb_sin(session, graph, node, uhs);
            break;
        case TBUOT_COS:
            return _tb_cos(session, graph, node, uhs);
            break;
        case TBUOT_TAN:
            return _tb_tan(session, graph, node, uhs);
            break;
        case TBUOT_TANH:
            return _tb_tanh(session, graph, node, uhs);
            break;
        case TBUOT_RELU:
            return _tb_relu(session, graph, node, uhs);
            break;
        case TBUOT_SOFTPLUS:
            return _tb_softplus(session, graph, node, uhs);
            break;
        case TBUOT_SIGMOID:
            return _tb_sigmoid(session, graph, node, uhs);
            break;
    }
}
