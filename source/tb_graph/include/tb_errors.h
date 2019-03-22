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
 * @file tb_errors.h
 * @author Soulaymen Chouri
 * @date March 16 2019
 * @brief File containing error handeling structures.
 * This file provides list of possible errors during graph processing
 */

#ifndef _TB_ERRORS_H_
#define _TB_ERRORS_H_


/**
 * \brief List of errors that can occure during computations
 */
typedef enum TBErrorType {
	/**
	 * \brief incompatible arguments given i.e device a vector by a matrix, 
	 *        like seriously .. what do you think you are doing?
	 */
	TBET_INCOMPATIBLE_ARGS_EXCEPTION = 0,
	
	/**
	 * \brief arguments' types are fine, yet their dimentions are not compatible.
	 */
	TBET_INCOMPATIBLE_DIMENTIONS_EXCEPTION,
	
	/**
	 * \brief double overflow exception
	 */
	TBET_OVERFLOW_EXCEPTION,
	
	/**
	 * \brief Variable not present in the current graph
	 */
	TBET_VARIABLE_DOES_NOT_EXIST,
	
	/**
	 * \brief Operation is not yet supported.
	 */
	TBET_OPERATION_NOT_IMPLEMENTED,
	
	/**
	 * \brief Did you just shoot yourself in the foot?
	 */
	TBET_DIVIDE_BY_ZERO,
	
	/**
	 * \brief Attempting to compute a variable node value when the graph instance is NULL, usually when calling `computeRawNode`
	 */
	TBET_NO_GRAPH_INSTANCE,
}TBErrorType;

#define TB_MAX_ERROR_TYPE TBET_NO_GRAPH_INSTANCE



typedef struct TBError {
	TBErrorType errorType;      /**< Which type of error has occured. */
	struct TBNode* faultyNode;  /**< Which node caused the error. */
    struct TBGraph* graph;      /**< Which Graph the exception occured */
	const char* message;              /**< Error message & description */
}TBError;

#endif
