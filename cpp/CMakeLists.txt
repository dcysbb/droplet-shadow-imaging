cmake_minimum_required(VERSION 3.16)
project(droplet_shadow_g4 LANGUAGES CXX)
find_package(Geant4 11.4 REQUIRED)
add_executable(droplet_g4 main.cpp)
target_compile_features(droplet_g4 PRIVATE cxx_std_17)
target_link_libraries(droplet_g4 PRIVATE ${Geant4_LIBRARIES})
target_include_directories(droplet_g4 PRIVATE ${Geant4_INCLUDE_DIRS})

