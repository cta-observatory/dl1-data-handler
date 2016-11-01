CC=g++
CFLAGS=-Wall -g
CFLAGS+=`pkg-config opencv --cflags`
CFLAGS+=`root-config --cflags`
LDFLAGS=`pkg-config opencv --libs`
LDFLAGS+=`root-config --libs`
imageExtractor:
	 ${CC} imageExtractor.C -o imageExtractor ${CFLAGS} ${LDFLAGS}
