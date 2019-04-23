#include <stdio.h>
#include <stdlib.h>

#include <ndarray.h>
#include <ndarray_std.h>

#include <tb_session.h>
#include <tb_graph.h>
#include <tb_factory.h>
#include <tb_ops.h>


void test(NDArray* lhs, NDArray* rhs){
    NDShape* lhsShape = lhs->shape;
    NDShape* rhsShape = rhs->shape;
    
    nda_debugShape(lhsShape);
    nda_debugShape(rhsShape);
    
    uint8_t res = nda_shapeCanBroadCast(lhsShape, rhsShape);
    
    if(!res){
        char msg[1024] = {0};
        char* lhsShapeInfo = nda_shapeToString(lhsShape);
        char* rhsShapeInfo = nda_shapeToString(rhsShape);
        snprintf(msg, 1024, "Cannot broadcast shapes %s and %s", lhsShapeInfo, rhsShapeInfo);
        
        free(lhsShapeInfo);
        free(rhsShapeInfo);
        
        printf("%s\n", msg);
        exit(-1);
    }
    
    printf("Can broadcast = %d\n", res);
    
    NDShape* biggerShape  = res==1?lhsShape:rhsShape;
    NDShape* smallerShape = res==1?rhsShape:lhsShape;
    
    uint64_t vshape_len = biggerShape->rank;
    uint64_t num_pads = biggerShape->rank-smallerShape->rank;
    
    uint64_t* array = calloc(vshape_len, sizeof(uint64_t));
    
    size_t i = 0;
    size_t j = 0;
    
    for(; i < num_pads; i++){
        array[i] = biggerShape->dims[i];
    }
    
    for(; i < vshape_len; i++,j++){
        if(biggerShape->dims[i] > smallerShape->dims[j]){
            array[i] = biggerShape->dims[i];
        }
        else {
            array[i] = smallerShape->dims[j];
        }
    }
    
    NDShape* vshape = nda_newShapeFromArray(vshape_len, array);
    nda_debugShape(vshape);
    
    NDArray* arr_res = nda_alloc(vshape);
    tb_float* arr = arr_res->data;
    
    size_t counter = 0;
    
    
    
    for(; counter<vshape->raw_len;counter++){
        size_t i = vshape->rank;
        uint64_t mul = vshape->raw_len;
        uint64_t counter_tmp = counter;
        uint64_t* index = calloc(vshape->rank, sizeof(uint64_t));
        for(; i !=0 ; i--){
            mul /= vshape->dims[vshape->rank - i];
            index[vshape->rank - i] = counter_tmp / mul;
            counter_tmp -= index[vshape->rank - i] * mul;
        }
        free(index);
        
        
        printf("index[%lld, %lld %lld] = %f\n", index[0], index[1], index[2], nda_vget(rhs, index, vshape));
        arr[counter] += nda_vget(lhs, index, vshape) + nda_vget(rhs, index, vshape);
    }
    nda_debugValue(arr_res);
}

void main_test(){
    
    NDArray* x = nda_linspace(0, 1, 3);
    NDArray* y = nda_linspace(0, 1, 3);
    
    nda_reshape(x, nda_newShape(2, 1, 3));
    nda_reshape(y, nda_newShape(2, 3, 1));
    
    nda_debugValue(x);
    nda_debugValue(y);
    
    test(x, y);
}


int main__(){
    
    NDArray* x = nda_linspace(0, 1, 16);
    NDArray* y = nda_linspace(0, 1, 2);
    
    nda_reshape(x, nda_newShape(3, 2, 4, 2));
    nda_reshape(y, nda_newShape(2, 1, 2));
    
    TBNode* n2 = tb_newBinaryOpNode(TBBOT_ADD, tb_newConstantNode(x), tb_newConstantNode(y));
    TBGraph* nestedGraph = tb_newGraph("nested", n2);
    
    TBNode* n1 = tb_newUnaryOpNode(TBUOT_RELU, tb_newGraphNode(nestedGraph, NULL));
    
    TBGraph* g = tb_newGraph("test", n1);
    
    TBResultNode* res = tb_runSession(NULL, g, NULL);
    
    nda_debugValue(res->value);
     
    return 0;
}
