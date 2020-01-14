.PHONY : clean

CXX=g++
#sources= *.cc
sources=$(shell echo *.cc)
objects=$(sources:.cc =.o)
target=librender_cpu_ref.so


rt:
	$(CXX) -std=c++17 -fPIC -g3 -lpng -o rt $(sources)


all: $(target)

clean:
	rm -f $(objects) $(target)

$(target) : $(objects)
	$(CXX) -std=c++17 -fPIC -lpng  $(object) -o $@ -shared

