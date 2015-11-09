#!/bin/bash

. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/initialize.sh "$@"

. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/resolve_dependency.sh \
  --submodules "ALL" "yong" "gtest"
. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/resolve_dependency.sh \
  --submodules "ALL" \
  --extra_cmake_defs "-DNASM=${CI_NASM_PATH}" "yong" "jpeg_turbo"
. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/resolve_dependency.sh \
  --submodules "ALL" "yong" "zlib"
. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/resolve_dependency.sh \
  --submodules "ALL" --deps "zlib" "yong" "png"
. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/resolve_dependency.sh \
  --submodules "ALL" --deps "jpeg_turbo zlib" "yong" "tiff"

  echo " ****** Initializing and dependency resolving complete, now running lib ci cmake_generate.sh ..."
. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/cmake_generate.sh \
  --extra_cmake_defs "-DHSLIB_COPY_TEST_DATA=1" --deps "gtest jpeg_turbo zlib png tiff"
