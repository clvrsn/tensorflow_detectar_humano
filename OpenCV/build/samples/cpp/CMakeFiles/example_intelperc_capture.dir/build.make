# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build

# Include any dependencies generated for this target.
include samples/cpp/CMakeFiles/example_intelperc_capture.dir/depend.make

# Include the progress variables for this target.
include samples/cpp/CMakeFiles/example_intelperc_capture.dir/progress.make

# Include the compile flags for this target's objects.
include samples/cpp/CMakeFiles/example_intelperc_capture.dir/flags.make

samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o: samples/cpp/CMakeFiles/example_intelperc_capture.dir/flags.make
samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o: ../samples/cpp/intelperc_capture.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o -c /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/cpp/intelperc_capture.cpp

samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.i"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/cpp/intelperc_capture.cpp > CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.i

samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.s"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/cpp/intelperc_capture.cpp -o CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.s

samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o.requires:

.PHONY : samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o.requires

samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o.provides: samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o.requires
	$(MAKE) -f samples/cpp/CMakeFiles/example_intelperc_capture.dir/build.make samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o.provides.build
.PHONY : samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o.provides

samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o.provides.build: samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o


# Object files for target example_intelperc_capture
example_intelperc_capture_OBJECTS = \
"CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o"

# External object files for target example_intelperc_capture
example_intelperc_capture_EXTERNAL_OBJECTS =

bin/cpp-example-intelperc_capture: samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o
bin/cpp-example-intelperc_capture: samples/cpp/CMakeFiles/example_intelperc_capture.dir/build.make
bin/cpp-example-intelperc_capture: /usr/lib/x86_64-linux-gnu/libGLU.so
bin/cpp-example-intelperc_capture: /usr/lib/x86_64-linux-gnu/libGL.so
bin/cpp-example-intelperc_capture: /usr/lib/x86_64-linux-gnu/libtbb.so
bin/cpp-example-intelperc_capture: 3rdparty/ippicv/ippicv_lnx/lib/intel64/libippicv.a
bin/cpp-example-intelperc_capture: lib/libopencv_shape.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_stitching.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_superres.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_videostab.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_viz.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_objdetect.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_photo.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_calib3d.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_features2d.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_flann.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_highgui.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_ml.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_videoio.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_imgcodecs.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_video.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_imgproc.so.3.2.0
bin/cpp-example-intelperc_capture: lib/libopencv_core.so.3.2.0
bin/cpp-example-intelperc_capture: /usr/lib/x86_64-linux-gnu/libGLU.so
bin/cpp-example-intelperc_capture: /usr/lib/x86_64-linux-gnu/libGL.so
bin/cpp-example-intelperc_capture: samples/cpp/CMakeFiles/example_intelperc_capture.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/cpp-example-intelperc_capture"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_intelperc_capture.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
samples/cpp/CMakeFiles/example_intelperc_capture.dir/build: bin/cpp-example-intelperc_capture

.PHONY : samples/cpp/CMakeFiles/example_intelperc_capture.dir/build

samples/cpp/CMakeFiles/example_intelperc_capture.dir/requires: samples/cpp/CMakeFiles/example_intelperc_capture.dir/intelperc_capture.cpp.o.requires

.PHONY : samples/cpp/CMakeFiles/example_intelperc_capture.dir/requires

samples/cpp/CMakeFiles/example_intelperc_capture.dir/clean:
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && $(CMAKE_COMMAND) -P CMakeFiles/example_intelperc_capture.dir/cmake_clean.cmake
.PHONY : samples/cpp/CMakeFiles/example_intelperc_capture.dir/clean

samples/cpp/CMakeFiles/example_intelperc_capture.dir/depend:
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/cpp /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp/CMakeFiles/example_intelperc_capture.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : samples/cpp/CMakeFiles/example_intelperc_capture.dir/depend

