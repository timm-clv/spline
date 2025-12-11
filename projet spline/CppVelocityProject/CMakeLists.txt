cmake_minimum_required(VERSION 3.10)
project(VelocityControllerCpp)

set(CMAKE_CXX_STANDARD 17)

# Inclusion des headers
include_directories(include)

# Définition des sources
set(SOURCES 
    src/main.cpp 
    src/VelocityController.cpp
)

# Exécutable
add_executable(VelocitySim ${SOURCES})