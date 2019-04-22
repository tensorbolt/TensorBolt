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
 * @brief File containing ndarray standard backend implementation aka tensors metadata.
 */

#ifndef _TB_NDARRAY_STD_BACKEND_H_
#define _TB_NDARRAY_STD_BACKEND_H_

#include <stdint.h>
#include <stdarg.h>

/**
 * \brief Tensor Shape
 */
typedef struct NDShape {
    uint64_t rank;     /**< Total number of dimentions */
    uint64_t* dims;    /**< Dimensions */
    uint64_t raw_len;  /**< Total number of elements */
    uint64_t* strides;
}NDShape;

/**
 * \brief Treats NDShape as a stack to pop elements, does not modify the original shape
 */
typedef struct NDShapeStack {
    NDShape* shape;       /**< Pointer to shape */
    uint64_t i;           /**< Stack Pointer */
}NDShapeStack;

/**
 * \brief Tensor data structure
 */
typedef struct NDArray {
    tb_float* data;  /**< Raw data as contigious array */
    NDShape* shape;  /**< Shape of the tensor */
}NDArray;

#endif
