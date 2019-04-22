cmake_minimum_required(VERSION 2.6)
project(unittest)

set (PROJECT_INCLUDE_DIR
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/../ndarray/include/
	${PROJECT_SOURCE_DIR}/../tb_graph/include/
	${PROJECT_SOURCE_DIR}/vendor/minunit/
)


set (PROJECT_SRCS
	${PROJECT_SOURCE_DIR}/source/test.c
)

include_directories(${PROJECT_INCLUDE_DIR})

add_executable(unittest 
	${PROJECT_SRCS}
)

target_link_libraries(unittest 
	ndarray
	tb_graph
)

install(TARGETS unittest RUNTIME DESTINATION bin)
