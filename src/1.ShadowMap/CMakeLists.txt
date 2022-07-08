

add_executable(1.ShadowMap main.cpp ../common.hpp)

target_link_libraries(1.ShadowMap PRIVATE CGUtils)

target_compile_features(1.ShadowMap PRIVATE cxx_std_20)