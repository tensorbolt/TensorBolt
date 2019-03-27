#include <stdio.h>
#include <stdlib.h>

#include <ndarray.h>

#include <tb_session.h>
#include <tb_graph.h>
#include <tb_factory.h>
#include <tb_ops.h>

int main(){
    NDArray* x = nda_linspace(0, 1, 10);
    nda_reshape(x, nda_newShape(3, 1, 2, 5));
    
    /**
    NDArray* y = nda_linspace(0, 1, 9);
    nda_reshape(y, nda_newShape(2, 3, 3));
    
    printf("%s\n", nda_shapeToString(x->shape));
    
    printf("can broadcast = %d\n", nda_shapeCanBroadCast(x->shape, y->shape));
    
    nda_debugValue(x);

    TBNode* n1 = tb_newUnaryOpNode(TBUOT_LOG, tb_newConstantNode(x));
    
    TBGraph* g = tb_newGraph("test", n1);
    
    TBResultNode* res = tb_runSession(NULL, g, NULL);
    
    nda_debugValue(res->value);
    */
    
    nda_debugValue(x);
    uint64_t index[] = {0, 1};
    
    uint64_t i = 0;
    
    for(; i < 2; i++){
        uint64_t j = 0;
        for(; j < 5; j++){
            uint64_t index[] = {0, 0, 3};
            index[1] = i;
            index[2] = j;
            
            printf("  %f, ", nda_get(x, index));
        }
        printf("\n");
    }
    
    tb_float val = nda_get(x, index);
    printf("value = %f\n", val);
    return 0;
}
