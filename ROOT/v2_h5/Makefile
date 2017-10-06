#CC=g++
CC=h5c++
CFLAGS=-Wall -g
CFLAGS+=`pkg-config opencv exiv2 --cflags`
CFLAGS+=`root-config --cflags`
CFLAGS+=-I/usr/local/hdf5/include
LDFLAGS=`pkg-config opencv exiv2 --libs`
LDFLAGS+=`root-config --libs`
LDFLAGS+=-L/usr/local/hdf5/lib

imageExtractor_v2:
	 ${CC} imageExtractor_v2.C -o imageExtractor_v2 ${CFLAGS} ${LDFLAGS}

.PHONY: clean
clean:
	rm -f imageExtractor_v2

.PHONY: all
all: clean imageExtractor_v2
