
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


MU_TEST(test_slice_01){
    NDArray* y = nda_linspace(0, 1, 8*4);
    nda_reshape(y, nda_newShape(3, 2, 8, 2));
    
    uint64_t index[] = {0,0, 3,6, 0,2};
    
    NDArray* x = nda_slice(y, index);
    
    uint64_t dims[] = {3, 2};
    tb_float gt[3][2] = {
        {0.19354839, 0.22580645},
        {0.25806452, 0.29032258},
        {0.32258065, 0.35483871},
    };
    
    
    mu_assert_int_eq(2, x->shape->rank);
    ASSERT_SHAPE_EQ(x->shape, dims);
    
    uint64_t idx[] = {0, 0};
    
    uint64_t i = 0;
    uint64_t j = 0;
    
    for(; i < x->shape->dims[0]; i++){
        for(j = 0; j < x->shape->dims[1]; j++){
            idx[0] = i;
            idx[1] = j;
            
            mu_assert_double_eq(gt[i][j], nda_get(x, idx));
            
        }
    }
}


MU_TEST(test_slice_02){
    NDArray* y = nda_linspace(0, 1, 8*4);
    nda_reshape(y, nda_newShape(3, 2, 8, 2));
    
    uint64_t index[] = {0,1, 3,6, 0,2};
    
    NDArray* x = nda_slice(y, index);
    
    uint64_t dims[] = {1, 3, 2};
    tb_float gt[1][3][2] = {{
        {0.19354839, 0.22580645},
        {0.25806452, 0.29032258},
        {0.32258065, 0.35483871},
    }};
    
    
    mu_assert_int_eq(3, x->shape->rank);
    ASSERT_SHAPE_EQ(x->shape, dims);
    
    uint64_t idx[] = {0, 0, 0};
    
    uint64_t i = 0;
    uint64_t j = 0;
    uint64_t k = 0;
    
    for(; i < x->shape->dims[0]; i++){
        for(j = 0; j < x->shape->dims[1]; j++){
            for(k = 0; k < x->shape->dims[2]; k++){
                idx[0] = i;
                idx[1] = j;
                idx[2] = k;
                
                mu_assert_double_eq(gt[i][j][k], nda_get(x, idx));
            }
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
    mu_assert_int_eq(2, res->value->shape->rank);
    
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

MU_TEST(test_transpose_1d){
    
    NDArray* x = nda_linspace(0, 1, 8);
    
    TBNode* n1 = tb_newTransposeOpNode(tb_newConstantNode(x), 0, 1);
    TBGraph* g = tb_newGraph("test", n1);
    
    TBResultNode* res = tb_runSession(NULL, g, NULL);
    
    uint64_t dims[] = {8, 1};
    uint64_t strides[] = {1, 8};
    
    mu_assert_int_eq(1, res->value->shape->rank);
    ASSERT_SHAPE_EQ(res->value->shape, dims);
    ASSERT_SHAPE_STRIDE_EQ(res->value->shape, strides);
    tb_float gt[] = {0.        , 0.14285714, 0.28571429, 0.42857143, 0.57142857,
        0.71428571, 0.85714286, 1.};
    
    uint64_t i =0;
    x = res->value;
    for(i = 0; i < 8; i++){
        uint64_t index[] = {0, 0};
        index[0] = i;
        index[1] = 0;
        
        mu_assert_double_eq(gt[i], nda_get(x, index));
    }
}

MU_TEST(test_transpose_dot1){
    
    NDArray* x = nda_linspace(0, 1, 8);
    nda_reshape(x, nda_newShape(2, 8, 1));
    
    
    NDArray* y = nda_linspace(0, 1, 8);
    nda_reshape(y, nda_newShape(2, 1, 8));
    
    TBNode* n0 = tb_newTransposeOpNode(tb_newConstantNode(x), 0, 1);
    TBNode* n1 = tb_newTransposeOpNode(tb_newConstantNode(y), 0, 1);
    TBNode* n2 = tb_newBinaryOpNode(TBBOT_DOT, n0, n1);
    TBGraph* g = tb_newGraph("test", n2);
    
    TBResultNode* res = tb_runSession(NULL, g, NULL);
    
    uint64_t dims[] = {1, 1};
    uint64_t strides[] = {1, 1};
    
    mu_assert_int_eq(2, res->value->shape->rank);
    ASSERT_SHAPE_EQ(res->value->shape, dims);
    ASSERT_SHAPE_STRIDE_EQ(res->value->shape, strides);
    
    uint64_t index[]={0,0};
    
    mu_assert_double_eq(2.857143, nda_get(res->value, index));
}


MU_TEST(test_transpose_dot2){
    
    NDArray* x = nda_linspace(0, 1, 8);
    nda_reshape(x, nda_newShape(2, 8, 1));
    
    
    NDArray* y = nda_linspace(0, 1, 8);
    nda_reshape(y, nda_newShape(2, 1, 8));
    
    TBNode* n0 = tb_newConstantNode(x);
    TBNode* n1 = tb_newConstantNode(y);
    TBNode* n2 = tb_newBinaryOpNode(TBBOT_DOT, n0, n1);
    TBGraph* g = tb_newGraph("test", n2);
    
    TBResultNode* res = tb_runSession(NULL, g, NULL);
    
    uint64_t dims[] = {8, 8};
    uint64_t strides[] = {8, 1};
    
    mu_assert_int_eq(2, res->value->shape->rank);
    ASSERT_SHAPE_EQ(res->value->shape, dims);
    ASSERT_SHAPE_STRIDE_EQ(res->value->shape, strides);
    
    uint64_t index[]={0,0};
    
    tb_float gt[8][8] = {
        {0.        , 0.        , 0.        , 0.        , 0.        ,
         0.        , 0.        , 0.        },
        {0.        , 0.02040816, 0.04081633, 0.06122449, 0.08163265,
         0.10204082, 0.12244898, 0.14285714},
        {0.        , 0.04081633, 0.08163265, 0.12244898, 0.16326531,
         0.20408163, 0.24489796, 0.28571429},
        {0.        , 0.06122449, 0.12244898, 0.18367347, 0.24489796,
         0.30612245, 0.36734694, 0.42857143},
        {0.        , 0.08163265, 0.16326531, 0.24489796, 0.32653061,
         0.40816327, 0.48979592, 0.57142857},
        {0.        , 0.10204082, 0.20408163, 0.30612245, 0.40816327,
         0.51020408, 0.6122449 , 0.71428571},
        {0.        , 0.12244898, 0.24489796, 0.36734694, 0.48979592,
         0.6122449 , 0.73469388, 0.85714286},
        {0.        , 0.14285714, 0.28571429, 0.42857143, 0.57142857,
         0.71428571, 0.85714286, 1.}
    };
    
    uint64_t i = 0, j = 0;
    
    for(; i < 8; i++){
        for(j=0; j<8; j++){
            index[0] = i;
            index[1] = j;
            
            mu_assert_double_eq(gt[i][j], nda_get(res->value, index));
        }
    }
}

MU_TEST(test_vec_mat_dot){
    
    NDArray* x = nda_linspace(0, 1, 8);
    
    
    NDArray* y = nda_linspace(0, 1, 8*4);
    nda_reshape(y, nda_newShape(2, 8, 4));
    
    TBNode* n0 = tb_newConstantNode(x);
    TBNode* n1 = tb_newConstantNode(y);
    TBNode* n2 = tb_newBinaryOpNode(TBBOT_DOT, n0, n1);
    TBGraph* g = tb_newGraph("test", n2);
    
    TBResultNode* res = tb_runSession(NULL, g, NULL);
    
    uint64_t dims[] = {4};
    uint64_t strides[] = {1};
    tb_float gt[] = {2.58064516, 2.70967742, 2.83870968, 2.96774194};
    
    
    mu_assert_int_eq(1, res->value->shape->rank);
    ASSERT_SHAPE_EQ(res->value->shape, dims);
    ASSERT_SHAPE_STRIDE_EQ(res->value->shape, strides);
    
    uint64_t i = 0;
    uint64_t index[] = {0};
    
    for(; i < 4; i++){
        index[0] = i;
        mu_assert_double_eq(gt[i], nda_get(res->value, index));
    }
    
}




MU_TEST_SUITE(nda_array_test) {
    MU_RUN_TEST(test_shape1);
    MU_RUN_TEST(test_shape2);
    MU_RUN_TEST(test_shape3);
    MU_RUN_TEST(test_shape4);
    MU_RUN_TEST(test_reshape1);
    MU_RUN_TEST(test_linspace);
    MU_RUN_TEST(test_slice_01);
    MU_RUN_TEST(test_slice_02);
}

MU_TEST_SUITE(tb_test) {
    MU_RUN_TEST(test_transpose_mult);
    MU_RUN_TEST(test_transpose_1d);
    MU_RUN_TEST(test_transpose_dot1);
    MU_RUN_TEST(test_transpose_dot2);
    MU_RUN_TEST(test_vec_mat_dot);
}

void runAllTests(){
    MU_RUN_SUITE(nda_array_test);
    MU_RUN_SUITE(tb_test);
    MU_REPORT();
}

int main(){
    printf("<TensorBolt & NDArray Test Units>\n\n");
    
    runAllTests();
    //test();
    return 0;
}
