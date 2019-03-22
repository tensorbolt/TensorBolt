project(ndarray)

set (PROJECT_INCLUDE_DIR
	/usr/local/include #for blas
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/vendor/container
)

set( EXTERNAL_DEPS
	${PROJECT_SOURCE_DIR}/vendor/container/vec.c
	${PROJECT_SOURCE_DIR}/vendor/container/map.c
)

set (PROJECT_SRCS
	${PROJECT_SOURCE_DIR}/source/ndarray.c
)
set (PROJECT_HEADERS
	${PROJECT_SOURCE_DIR}/include/ndarray.h
)

include_directories(${PROJECT_INCLUDE_DIR})

add_library(ndarray
#	${EXTERNAL_DEPS}
	${PROJECT_SRCS}
	${PROJECT_HEADERS}
)
# 
install(TARGETS ndarray
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
