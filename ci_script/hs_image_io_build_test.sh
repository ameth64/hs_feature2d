#!/bin/bash

. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/initialize.sh "$@"
. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/build_test.sh \
  "stream_io_utest"
. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/build_test.sh \
  "whole_io_utest"
