

add_executable(2.PRT main.cpp ../common.hpp)

target_link_libraries(2.PRT PRIVATE CGUtils)

target_compile_features(2.PRT PRIVATE cxx_std_20)