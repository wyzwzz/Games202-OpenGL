

add_executable(3.SSRT main.cpp ../common.hpp)

target_link_libraries(3.SSRT PRIVATE CGUtils)

target_compile_features(3.SSRT PRIVATE cxx_std_20)

target_include_directories(3.SSRT PRIVATE ${PROJECT_SOURCE_DIR}/ext/cy)