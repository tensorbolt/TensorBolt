



#include <tb_session.h>
#include <tb_graph.h>

/* * * * * * * *
 * Session API *
 * * * * * * * */ 

TBGraphSession* tb_createLocalCPUSession(TBGraph* graph, TBGraphNodeParam** params){
    TBGraphSession* session = calloc(1, sizeof(TBGraphSession));
    session->device = "/local/cpu:0";
    session->graph = graph;

    return session;
}

// TODO
TBResultNode* tb_runSession(TBGraphSession* session){
    return NULL;
}

// TODO
void tb_freeSession(TBGraphSession* session){

}
