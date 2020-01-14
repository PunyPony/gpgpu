cmake_minimum_required(VERSION 3.5)

project(cpu_library)

############################################################
# Create a library
############################################################

#Generate the shared library from the library sources
add_library(render_cpu_ref SHARED 
    src/render.cc
)


target_compile_features(render_cpu_ref PUBLIC cxx_std_17)
add_library(cpu::library ALIAS render_cpu_ref)

target_include_directories(render_cpu_ref
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
)

############################################################
# Create an executable
############################################################

# Add an executable with the above sources

#add_executable(cpu_binary
#    src/test.cc
#)

#target_compile_features(cpu_binary PUBLIC cxx_std_17)
# link the new hello_library target with the hello_binary target
#target_link_libraries( cpu_binary
#    PRIVATE 
#        cpu::library
#)
