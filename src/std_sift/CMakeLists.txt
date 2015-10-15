set(STD_SIFT_SOURCES
  "image_helper.cpp"
  "run_main.cpp"
  )

hslib_add_library(std_sift
  SOURCES ${STD_SIFT_SOURCES})

yong_add_dependence(hs_feature2d std_sift
  DEPENDENT_PROJECT jpeg_turbo
    DEPENDENT_LIBS jpeg_turbo
  DEPENDENT_PROJECT tiff
    DEPENDENT_LIBS tiff
  DEPENDENT_PROJECT zlib
    DEPENDENT_LIBS zlib
  DEPENDENT_PROJECT hs_image_io
    DEPENDENT_LIBS stream_io whole_io
  )

hslib_add_library_end(std_sift)
