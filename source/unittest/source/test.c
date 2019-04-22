
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>

#include "minunit.h"
#include "ndarray.h"
#include "ndarray_std.h"


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

MU_TEST_SUITE(nda_array_test) {
    MU_RUN_TEST(test_shape1);
    MU_RUN_TEST(test_shape2);
    MU_RUN_TEST(test_shape3);
    MU_RUN_TEST(test_shape4);
}

void runAllTests(){
    MU_RUN_SUITE(nda_array_test);
    MU_REPORT();
}


int main(){
    printf("<TensorBolt & NDArray Test Units>\n\nRunning tests\n");
    
    runAllTests();

    return 0;
}
