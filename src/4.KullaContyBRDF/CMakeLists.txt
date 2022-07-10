

add_executable(4.KullaContyBRDF main.cpp ../common.hpp)

target_link_libraries(4.KullaContyBRDF PRIVATE CGUtils)

target_compile_features(4.KullaContyBRDF PRIVATE cxx_std_20)