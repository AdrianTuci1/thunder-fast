cmake_minimum_required(VERSION 3.20)
project(thunder_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TF_RUNTIME_GGML_SOURCE "Path to a local ggml checkout (overrides FetchContent)" "")

# Fetch ggml (the tensor/backend library). By default we vendor it; set the option to
# use a pinned local checkout for reproducible builds.
if(TF_RUNTIME_GGML_SOURCE)
    add_subdirectory(${TF_RUNTIME_GGML_SOURCE} ggml)
else()
    include(FetchContent)
    FetchContent_Declare(
        ggml
        GIT_REPOSITORY https://github.com/ggml-org/ggml.git
        GIT_TAG        b4227          # pin; adjust to a tested commit at build time
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(ggml)
endif()

# The runtime engine links ggml (tensors + backends) and the diffusion core + tokenizer.
add_library(runtime_core STATIC
    src/engine/model.cpp
    src/tokenizer/tokenizer.cpp
)
target_include_directories(runtime_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(runtime_core PUBLIC ggml)

add_executable(runtime src/main.cpp)
target_link_libraries(runtime PRIVATE runtime_core)
set_target_properties(runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Diffusion-core standalone test (no ggml) — runs regardless of backend.
add_executable(test_diffusion tests/test_diffusion.cpp)
target_include_directories(test_diffusion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

enable_testing()
add_test(NAME diffusion_core COMMAND ${CMAKE_BINARY_DIR}/test_diffusion)
