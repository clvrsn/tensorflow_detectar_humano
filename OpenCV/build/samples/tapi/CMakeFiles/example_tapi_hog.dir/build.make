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
include samples/tapi/CMakeFiles/example_tapi_hog.dir/depend.make

# Include the progress variables for this target.
include samples/tapi/CMakeFiles/example_tapi_hog.dir/progress.make

# Include the compile flags for this target's objects.
include samples/tapi/CMakeFiles/example_tapi_hog.dir/flags.make

samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o: samples/tapi/CMakeFiles/example_tapi_hog.dir/flags.make
samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o: ../samples/tapi/hog.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/tapi && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_tapi_hog.dir/hog.cpp.o -c /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/tapi/hog.cpp

samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_tapi_hog.dir/hog.cpp.i"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/tapi && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/tapi/hog.cpp > CMakeFiles/example_tapi_hog.dir/hog.cpp.i

samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_tapi_hog.dir/hog.cpp.s"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/tapi && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/tapi/hog.cpp -o CMakeFiles/example_tapi_hog.dir/hog.cpp.s

samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o.requires:

.PHONY : samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o.requires

samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o.provides: samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o.requires
	$(MAKE) -f samples/tapi/CMakeFiles/example_tapi_hog.dir/build.make samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o.provides.build
.PHONY : samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o.provides

samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o.provides.build: samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o


# Object files for target example_tapi_hog
example_tapi_hog_OBJECTS = \
"CMakeFiles/example_tapi_hog.dir/hog.cpp.o"

# External object files for target example_tapi_hog
example_tapi_hog_EXTERNAL_OBJECTS =

bin/tapi-example-hog: samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o
bin/tapi-example-hog: samples/tapi/CMakeFiles/example_tapi_hog.dir/build.make
bin/tapi-example-hog: /usr/lib/x86_64-linux-gnu/libGLU.so
bin/tapi-example-hog: /usr/lib/x86_64-linux-gnu/libGL.so
bin/tapi-example-hog: /usr/lib/x86_64-linux-gnu/libtbb.so
bin/tapi-example-hog: 3rdparty/ippicv/ippicv_lnx/lib/intel64/libippicv.a
bin/tapi-example-hog: lib/libopencv_video.so.3.2.0
bin/tapi-example-hog: lib/libopencv_objdetect.so.3.2.0
bin/tapi-example-hog: lib/libopencv_calib3d.so.3.2.0
bin/tapi-example-hog: lib/libopencv_features2d.so.3.2.0
bin/tapi-example-hog: lib/libopencv_highgui.so.3.2.0
bin/tapi-example-hog: lib/libopencv_videoio.so.3.2.0
bin/tapi-example-hog: lib/libopencv_imgcodecs.so.3.2.0
bin/tapi-example-hog: lib/libopencv_imgproc.so.3.2.0
bin/tapi-example-hog: lib/libopencv_flann.so.3.2.0
bin/tapi-example-hog: lib/libopencv_ml.so.3.2.0
bin/tapi-example-hog: lib/libopencv_core.so.3.2.0
bin/tapi-example-hog: samples/tapi/CMakeFiles/example_tapi_hog.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/tapi-example-hog"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/tapi && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_tapi_hog.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
samples/tapi/CMakeFiles/example_tapi_hog.dir/build: bin/tapi-example-hog

.PHONY : samples/tapi/CMakeFiles/example_tapi_hog.dir/build

samples/tapi/CMakeFiles/example_tapi_hog.dir/requires: samples/tapi/CMakeFiles/example_tapi_hog.dir/hog.cpp.o.requires

.PHONY : samples/tapi/CMakeFiles/example_tapi_hog.dir/requires

samples/tapi/CMakeFiles/example_tapi_hog.dir/clean:
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/tapi && $(CMAKE_COMMAND) -P CMakeFiles/example_tapi_hog.dir/cmake_clean.cmake
.PHONY : samples/tapi/CMakeFiles/example_tapi_hog.dir/clean

samples/tapi/CMakeFiles/example_tapi_hog.dir/depend:
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/tapi /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/tapi /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/tapi/CMakeFiles/example_tapi_hog.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : samples/tapi/CMakeFiles/example_tapi_hog.dir/depend

