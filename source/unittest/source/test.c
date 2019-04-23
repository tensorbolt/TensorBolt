
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>

#include "minunit.h"
#include "ndarray.h"
#include "ndarray_std.h"

#include <tb_session.h>
#include <tb_graph.h>
#include <tb_factory.h>
#include <tb_ops.h>

#define ASSERT_SHAPE_EQ(shape, values)\
{\
uint64_t i = 0;\
\
for(;i < shape->rank;i++){\
mu_assert_int_eq(values[i], shape->dims[i]);\
}\
}

#define ASSERT_SHAPE_STRIDE_EQ(shape, values)\
{\
uint64_t i = 0;\
\
for(;i < shape->rank;i++){\
mu_assert_int_eq(values[i], shape->strides[i]);\
}\
}

MU_TEST(test_shape1){
    NDShape* shape = nda_newShape(3, 1, 2, 3);
    uint64_t dims[] = {1, 2, 3};
    mu_assert_int_eq(3, shape->rank);
    ASSERT_SHAPE_EQ(shape, dims);
}

MU_TEST(test_shape2){
    uint64_t dims[] = {1, 2, 3};
    NDShape* shape = nda_newShapeFromArrayCopy(3, dims);
    mu_assert_int_eq(3, shape->rank);
    ASSERT_SHAPE_EQ(shape, dims);
}

MU_TEST(test_shape3){
    uint64_t* dims = calloc(3, sizeof(uint64_t));
    dims[0] = 1;
    dims[1] = 2;
    dims[2] = 3;
    
    uint64_t dims2[] = {1, 2, 3};
    
    NDShape* shape = nda_newShapeFromArray(3, dims);
    mu_assert_int_eq(3, shape->rank);
    ASSERT_SHAPE_EQ(shape, dims2);
}

MU_TEST(test_shape4){
    NDShape* shape = nda_newShape(4, 4, 3, 2, 3);
    uint64_t dims[] = {4, 3, 2, 3};
    uint64_t strides[] = {18, 6, 3, 1};
    mu_assert_int_eq(4, shape->rank);
    ASSERT_SHAPE_EQ(shape, dims);
    ASSERT_SHAPE_STRIDE_EQ(shape, strides);
}

MU_TEST(test_reshape1){
    NDArray* x = nda_linspace(0, 1, 6);
    mu_assert_int_eq(1, x->shape->rank);
    mu_assert_int_eq(6, x->shape->dims[0]);
    
    nda_reshape(x, nda_newShape(2, 2, 3));
    mu_assert_int_eq(2, x->shape->rank);
    uint64_t dims[] = {2, 3};
    
    ASSERT_SHAPE_EQ(x->shape, dims);
}

MU_TEST(test_linspace){
    NDArray* x = nda_linspace(0, 1, 6);
    nda_reshape(x, nda_newShape(2, 2, 3));

    size_t i = 0;
    size_t j = 0;

    tb_float values[2][3] = {{0. , 0.2, 0.4}, {0.6, 0.8, 1.}};
    
    for (; i < 2; i++){
        for(j = 0; j < 3; j++){
            uint64_t index[] = {0, 0};
            index[0] = i;
            index[1] = j;
            
            mu_assert_double_eq(values[i][j], nda_get(x, index));
        }
    }
}


MU_TEST(test_transpose_mult){
    NDArray* x = nda_linspace(0, 1, 16);
    nda_reshape(x, nda_newShape(2, 4, 4));
    
    NDArray* y = nda_linspace(0, 1, 4);
    
    tb_float gt[4][4] = {{0.        , 0.08888889, 0.35555556, 0.8       },
        {0.        , 0.11111111, 0.4       , 0.86666667},
        {0.        , 0.13333333, 0.44444444, 0.93333333},
        {0.        , 0.15555556, 0.48888889, 1.}};
    
    
    TBNode* n1 = tb_newTransposeOpNode(tb_newConstantNode(x), 0, 1);
    TBNode* n2 = tb_newBinaryOpNode(TBBOT_MULT, n1, tb_newConstantNode(y));
    
    TBGraph* g = tb_newGraph("test", n2);
    
    TBResultNode* res = tb_runSession(NULL, g, NULL);
    
    uint64_t i =0, j =0;
    x = res->value;
    for(i=0; i<4;i++){
        for(j = 0; j < 4; j++){
            uint64_t index[] = {0, 0};
            index[0] = i;
            index[1] = j;
            
            mu_assert_double_eq(gt[i][j], nda_get(x, index));
        }
    }
}

MU_TEST_SUITE(nda_array_test) {
    MU_RUN_TEST(test_shape1);
    MU_RUN_TEST(test_shape2);
    MU_RUN_TEST(test_shape3);
    MU_RUN_TEST(test_shape4);
    MU_RUN_TEST(test_reshape1);
    MU_RUN_TEST(test_linspace);
}

MU_TEST_SUITE(tb_test) {
    MU_RUN_TEST(test_transpose_mult);
}

void runAllTests(){
    MU_RUN_SUITE(nda_array_test);
    MU_RUN_SUITE(tb_test);
    MU_REPORT();
}


int main(){
    printf("<TensorBolt & NDArray Test Units>\n\n");
    
    runAllTests();
    return 0;
}
