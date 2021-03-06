project(tb_graph)

set (PROJECT_INCLUDE_DIR
	/usr/local/include #for blas
	${PROJECT_SOURCE_DIR}/../ndarray/include
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/vendor/container
)

set( EXTERNAL_DEPS
	${PROJECT_SOURCE_DIR}/vendor/container/vec.c
	${PROJECT_SOURCE_DIR}/vendor/container/map.c
)

set (PROJECT_SRCS
	${PROJECT_SOURCE_DIR}/source/main.c
	${PROJECT_SOURCE_DIR}/source/tb_graph.c
	${PROJECT_SOURCE_DIR}/source/tb_factory.c
	${PROJECT_SOURCE_DIR}/source/tb_session_cpu.c
	${PROJECT_SOURCE_DIR}/source/tb_ops_cpu.c
	${PROJECT_SOURCE_DIR}/source/tb_autograd.c
)

set (PROJECT_HEADERS
	${PROJECT_SOURCE_DIR}/include/tb_graph.h
	${PROJECT_SOURCE_DIR}/include/tb_factory.h
	${PROJECT_SOURCE_DIR}/include/tb_session.h
	${PROJECT_SOURCE_DIR}/include/tb_session_cpu.h
	${PROJECT_SOURCE_DIR}/include/tb_operation.h
	${PROJECT_SOURCE_DIR}/include/tb_errors.h
	${PROJECT_SOURCE_DIR}/include/tb_ops.h
	${PROJECT_SOURCE_DIR}/include/tb_autograd.h
)

add_library(tb_graph
	${EXTERNAL_DEPS}
	${PROJECT_SRCS}
	${PROJECT_HEADERS}
)

find_package(OpenBLAS)


include_directories(
	${PROJECT_INCLUDE_DIR} 
	${OpenBLAS_INCLUDE_DIR}
)

target_link_libraries(tb_graph 
	ndarray
	${OpenBLAS_LIB}
)

install(TARGETS tb_graph
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

