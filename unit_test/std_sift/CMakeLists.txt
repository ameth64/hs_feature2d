set(STD_SIFT_UTEST_SOURCES
  "${PROJECT_SOURCE_DIR}/unit_test/test_main.cpp"
  "test_std_sift.cpp"
  )
hslib_add_utest(std_sift SOURCES ${STD_SIFT_UTEST_SOURCES})
yong_add_dependence(hs_feature2d std_sift_utest
#待测试项目的库
  DEPENDENT_LOCAL_LIBS std_sift
  DEPENDENT_PROJECT jpeg_turbo
    DEPENDENT_LIBS jpeg_turbo
  DEPENDENT_PROJECT png
    DEPENDENT_LIBS png
  DEPENDENT_PROJECT tiff
    DEPENDENT_LIBS tiff
  DEPENDENT_PROJECT zlib
    DEPENDENT_LIBS zlib
  DEPENDENT_PROJECT hs_image_io
    DEPENDENT_LIBS stream_io whole_io
  )
hslib_add_mkl_dep(std_sift_utest)
hslib_add_utest_end(std_sift)
