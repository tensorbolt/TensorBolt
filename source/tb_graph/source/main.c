#include <stdio.h>
#include <stdlib.h>

#include <ndarray.h>

#include <tb_session.h>
#include <tb_graph.h>
#include <tb_factory.h>

int main(){
    NDArray* x = nda_linspace(0, 1, 10);
    nda_debugValue(x);

    NDArray* y = nda_copy(x);
    nda_reshape(y, nda_newShape(2, 5, 2));
    nda_debugValue(y);
	
	nda_free(x);
	nda_free(y);
	
	TBNode* n = tb_newVarNode("x2+");

    TBGraphSession* session = tb_createLocalCPUSession(NULL, NULL);

    return 0;
}
