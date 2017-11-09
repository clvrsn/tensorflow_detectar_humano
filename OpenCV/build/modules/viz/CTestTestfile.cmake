# CMake generated Testfile for 
# Source directory: /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/modules/viz
# Build directory: /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/modules/viz
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_viz "/home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/bin/opencv_test_viz" "--gtest_output=xml:opencv_test_viz.xml")
set_tests_properties(opencv_test_viz PROPERTIES  LABELS "Main;opencv_viz;Accuracy" WORKING_DIRECTORY "/home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/test-reports/accuracy")
