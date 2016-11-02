CC=g++
CFLAGS=-Wall -g
CFLAGS+=`pkg-config opencv exiv2 --cflags`
CFLAGS+=`root-config --cflags`
LDFLAGS=`pkg-config opencv exiv2 --libs`
LDFLAGS+=`root-config --libs`
imageExtractor:
	 ${CC} imageExtractor.C -o imageExtractor ${CFLAGS} ${LDFLAGS}

.PHONY: clean
clean:
	rm -f imageExtractor
