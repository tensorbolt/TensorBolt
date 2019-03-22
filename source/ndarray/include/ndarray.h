/****************************************************************************
 * Copyright (C) 2019 by Soulaymen Chouri                                   *
 *                                                                          *
 * This file is part of TensorBolt.                                         *
 *                                                                          *
 * What follows is the Modified BSD License.                                *
 *     See also http://www.opensource.org/licenses/BSD-3-Clause             *
 * Copyright (c) 2019, Soulaymen Chouri. All rights reserved.               *
 * Redistribution and use in source and binary forms, with or without       *
 * modification, are permitted provided that the following conditions       *
 * are met:                                                                 *
 *                                                                          *
 *      1. Redistributions of source code must retain the above copyright   *
 *         notice, this list of conditions and the following disclaimer.    *
 *                                                                          *
 *      2. Redistributions in binary form must reproduce the above          *
 *         copyright notice, this list of conditions and the following      *
 *         disclaimer in the documentation and/or other materials provided  *
 *         with the distribution.                                           *
 *                                                                          *
 *      3. Neither the name of the author nor the names of other            *
 *         contributors may be used to endorse or promote products derived  *
 *         from this software without specific prior written permission.    *
 *                                                                          *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS "AS IS" AND ANY EXPRESS OR      *
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED           *
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE   *
 * DISCLAIMED. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT,      *
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES       *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR       *
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)       *
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,      *
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING    *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE       *
 * POSSIBILITY OF SUCH DAMAGE.                                              *
 ****************************************************************************/

/**
 * @file ndarray.h
 * @author Soulaymen Chouri
 * @date March 16 2019
 * @brief File containing ndarray aka tensors metadata.
 */

#ifndef _TB_NDARRAY_H_
#define _TB_NDARRAY_H_

#include <stdint.h>
#include <stdarg.h>

/*
 * change this to double if you need
 */
typedef float tb_float;

/**
 * \brief Tensor Shape
 */
typedef struct NDShape {
    uint64_t rank;  /**< Total number of dimentions */
    uint64_t* dims; /**< Dimensions */
}NDShape;

/**
 * \brief Treats NDShape as a stack to pop elements, does not modify the original shape
 */
typedef struct NDShapeStack {
    NDShape* shape;       /**< Pointer to shape */
    uint64_t i;           /**< Stack Pointer */
}NDShapeStack;

/**
 * \brief Unline other factory pattern object, the stack is usually allocated on the stack (in contrast to heap)
 * So we do need dynamically allocate it, this function only initializes the values of a fresh allocated stack
 * \param [in/out] stack Allocated stack to initialize
 * \param [in] shape Shape to bind to the stack
 */
void nda_ShapeStackInit(NDShapeStack* stack, NDShape* shape);

/**
 * \brief Checks if a shape stack can still popped further more
 * \param[in] stack Stack to check
 * \return Boolean, true if the stack can be popped further more, false otherwise.
 */
uint8_t nda_ShapeStackCanPop(NDShapeStack* stack);

/**
 * \brief Pops the next element in the stack, does not modify the shape but updates the index in the stack
 * \param[in/out] stack Stack from which to pop the element
 * \return Dimension of the shape in the stack index
 */
uint64_t nda_ShapeStackPop(NDShapeStack* stack);

/**
 * Array location
 */
typedef enum NDArrayLocation {
    NDA_LOC_HOST_MEM=0,   /**< Array is located on the host memory (RAM) */
    NDA_LOC_DEVICE_MEM=0, /**< Array is located on the device memory (global memory) */
}NDArrayLocation;

/**
 * \brief Tensor data structure
 */
typedef struct NDArray {
#ifdef TB_USE_OPENCL
    NDArrayLocation loc;
	CCLBuffer* buf;
#endif

    tb_float* data;  /**< Raw data as contigious array */
    NDShape* shape;  /**< Shape of the tensor */
}NDArray;

/**
 * Creates a new shape from the given elements dimensions
 * \param[in] rank number of dimensions
 * \param[in] ... list of dimensions (variadic parameters)
 */
NDShape* nda_newShape(uint64_t rank, ...);

/**
 * \brief Prints tensor shape to stdout
 * \param[in] shape TensorShape to display
 */
void nda_debugShape(NDShape* shape);

/**
 * \brief Creates and Allocates a shape with the same properties of the given one
 * \param shape Shape to copy
 * \return New allocated shape
 */
NDShape* nda_copyShape(NDShape* shape);

/**
 * \brief Prints tensor value to stdout
 * \param[in] tensor Tensor to display
 */
void nda_debugValue(NDArray* tensor);

/**
 * \brief Calculate the total number of elements in a tensor shape
 * \param[in] shape Tensor shape to process
 * \return Total number of elements in the array (Product of dimensions)
 */
uint64_t nda_getTotalSize(NDShape* shape);

/**
 * \brief Creates an empty zeroed tensor
 * \param shape initial shape
 * \return 0 initialized tensor
 */
NDArray* nda_alloc(NDShape* shape);

/**
 * \brief Return evenly spaced numbers over a specified interval.
 * \param[in] a start element
 * \param[in] b end element
 * \param[in] n number of elements
 * \return evenly spaced numbers over the specified interval.
 */
NDArray* nda_linspace(tb_float a, tb_float b, uint64_t n);

/**
 * \brief copies an existent tensor, memory must be explicitly freed.
 * \param[in] x tensor to copy
 * \return copy of x, must be explicitly freed
 */
NDArray* nda_copy(NDArray* x);

/**
 * \brief reshape an ndarray, old shape is freed.
 * \param[in/out] x ndarray to reshape
 * \param[in] shape new shape
 */
void nda_reshape(NDArray* x, NDShape* shape);

/**
 * \brief frees an NDArray alongside its shape
 * \param [in/out] array data to free
 */
void nda_free(NDArray* array);

#endif
