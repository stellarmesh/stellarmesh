cmake_minimum_required(VERSION 3.1)
project(CheckModel)

# 1. Setup paths
set(CONDA_ENV_PATH $ENV{CONDA_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${CONDA_ENV_PATH})

# 2. Find MOAB
find_package(MOAB REQUIRED)

# 3. Define executable
add_executable(check_model check_model.cpp)

# 4. Hardcode RPATH into the target
# We use the absolute path to the lib directory in your Pixi env
set_target_properties(check_model PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CONDA_ENV_PATH}/lib"
)

# 5. Link and Include
target_include_directories(check_model PUBLIC ${MOAB_INCLUDE_DIRS})
target_link_libraries(check_model PUBLIC ${MOAB_LIBRARIES})