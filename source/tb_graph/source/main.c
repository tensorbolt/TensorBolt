#include <stdio.h>
#include <stdlib.h>

#include <ndarray.h>

#include <tb_session.h>
#include <tb_graph.h>
#include <tb_factory.h>
#include <tb_ops.h>

int main(){
    NDArray* x = nda_linspace(0, 1, 10);
    nda_debugValue(x);

    TBNode* n1 = tb_newUnaryOpNode(TBUOT_LOG, tb_newConstantNode(x));
    
    TBGraph* g = tb_newGraph("test", n1);
    
    TBResultNode* res = tb_runSession(NULL, g, NULL);
    
    nda_debugValue(res->value);
    
    return 0;
}
