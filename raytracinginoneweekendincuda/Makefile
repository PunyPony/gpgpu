CXX = g++
sources= *.cc
objects=$(source:.cc =.o)
rt: main.cc camera.h hitable.h hitable_list.h material.h ray.h sphere.h vec3.h
	$(CXX)  -std=c++17 -g3 -lpng -o rt main.cc

libb:$(objects)
	$(CXX) -shared -std=c++17  -lpng -o libfoo.so


out.ppm: rt
	rm -f out.ppm
	time ./rt > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

clean:
	rm -f rt rt.o out.ppm out.jpg
