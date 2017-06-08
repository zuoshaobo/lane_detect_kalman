CXX=g++
CXXFLAGS=-I/home/pub/anaconda2/include/ -I/home/pub/anaconda2/include/opencv/
LDFLAGS=-lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_video -lopencv_objdetect -lopencv_ml -lopencv_contrib
exe:=$(patsubst %.cpp,%,$(wildcard *.cpp))
objs:=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
all:$(exe)
$(exe):%:%.cpp
	$(CXX) $< -o $@ $(LDFLAGS) $(CXXFLAGS)
clean:
	rm -fr *.o $(exe)
