cmake_minimum_required(VERSION 3.17)

project(Games202 LANGUAGES CXX C)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(BUILD_ENABLE_OPENGL ON)
add_subdirectory(ext/CGUtils)


add_subdirectory(src/1.ShadowMap)
add_subdirectory(src/2.PRT)
add_subdirectory(src/3.SSRT)
add_subdirectory(src/4.KullaContyBRDF)