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
include samples/cpp/CMakeFiles/tutorial_transformations.dir/depend.make

# Include the progress variables for this target.
include samples/cpp/CMakeFiles/tutorial_transformations.dir/progress.make

# Include the compile flags for this target's objects.
include samples/cpp/CMakeFiles/tutorial_transformations.dir/flags.make

samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o: samples/cpp/CMakeFiles/tutorial_transformations.dir/flags.make
samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o: ../samples/cpp/tutorial_code/viz/transformations.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o -c /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/cpp/tutorial_code/viz/transformations.cpp

samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.i"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/cpp/tutorial_code/viz/transformations.cpp > CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.i

samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.s"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/cpp/tutorial_code/viz/transformations.cpp -o CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.s

samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o.requires:

.PHONY : samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o.requires

samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o.provides: samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o.requires
	$(MAKE) -f samples/cpp/CMakeFiles/tutorial_transformations.dir/build.make samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o.provides.build
.PHONY : samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o.provides

samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o.provides.build: samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o


# Object files for target tutorial_transformations
tutorial_transformations_OBJECTS = \
"CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o"

# External object files for target tutorial_transformations
tutorial_transformations_EXTERNAL_OBJECTS =

bin/cpp-tutorial-transformations: samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o
bin/cpp-tutorial-transformations: samples/cpp/CMakeFiles/tutorial_transformations.dir/build.make
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libGLU.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libGL.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libtbb.so
bin/cpp-tutorial-transformations: 3rdparty/ippicv/ippicv_lnx/lib/intel64/libippicv.a
bin/cpp-tutorial-transformations: lib/libopencv_shape.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_stitching.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_superres.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_videostab.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_viz.so.3.2.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libz.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libjpeg.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libpng.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libtiff.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkFiltersTexture-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libfreetype.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkIOExport-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkRenderingGL2PS-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libgl2ps.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkRenderingLabel-6.3.so.6.3.0
bin/cpp-tutorial-transformations: lib/libopencv_objdetect.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_photo.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_calib3d.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_features2d.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_flann.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_highgui.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_ml.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_videoio.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_imgcodecs.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_video.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_imgproc.so.3.2.0
bin/cpp-tutorial-transformations: lib/libopencv_core.so.3.2.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libGLU.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkIOImage-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkIOCore-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkmetaio-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libz.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libSM.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libICE.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libX11.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libXext.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libXt.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkalglib-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtksys-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libvtkftgl-6.3.so.6.3.0
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libGL.so
bin/cpp-tutorial-transformations: /usr/lib/x86_64-linux-gnu/libfreetype.so
bin/cpp-tutorial-transformations: samples/cpp/CMakeFiles/tutorial_transformations.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/cpp-tutorial-transformations"
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tutorial_transformations.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
samples/cpp/CMakeFiles/tutorial_transformations.dir/build: bin/cpp-tutorial-transformations

.PHONY : samples/cpp/CMakeFiles/tutorial_transformations.dir/build

samples/cpp/CMakeFiles/tutorial_transformations.dir/requires: samples/cpp/CMakeFiles/tutorial_transformations.dir/tutorial_code/viz/transformations.cpp.o.requires

.PHONY : samples/cpp/CMakeFiles/tutorial_transformations.dir/requires

samples/cpp/CMakeFiles/tutorial_transformations.dir/clean:
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp && $(CMAKE_COMMAND) -P CMakeFiles/tutorial_transformations.dir/cmake_clean.cmake
.PHONY : samples/cpp/CMakeFiles/tutorial_transformations.dir/clean

samples/cpp/CMakeFiles/tutorial_transformations.dir/depend:
	cd /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/samples/cpp /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp /home/cleverson/pessoal/cleverson/workspace/mestrado/detector_humanos/OpenCV/build/samples/cpp/CMakeFiles/tutorial_transformations.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : samples/cpp/CMakeFiles/tutorial_transformations.dir/depend

