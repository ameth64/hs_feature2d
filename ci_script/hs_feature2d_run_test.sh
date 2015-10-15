#!/bin/bash

. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/initialize.sh "$@"
. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/run_test.sh \
  "unit_test/stream_io" "stream_io_utest"
. ${CI_PROJECT_DIR}/ci_script/hslib_ci_script/run_test.sh \
  "unit_test/whole_io" "whole_io_utest"
