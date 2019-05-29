src=./src
build=./build

# compiler
CC=g++
CFLAGS=-c -Wall -std=c++11 -I.

.PHONY: all
all: kernelNodeEmb

kernelNodeEmb: main.o Graph.o Node.o Model.o Unigram.o Vocabulary.o
	$(CC) ${build}/main.o ${build}/Graph.o ${build}/Node.o ${build}/Model.o ${build}/Unigram.o ${build}/Vocabulary.o -o kernelNodeEmb

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp -o ${build}/main.o

Graph.o: ${src}/Graph.cpp
	$(CC) $(CFLAGS) ${src}/Graph.cpp -o ${build}/Graph.o

Node.o: ${src}/Node.cpp
	$(CC) $(CFLAGS) ${src}/Node.cpp -o ${build}/Node.o

Model.o: ${src}/Model.cpp
	$(CC) $(CFLAGS) ${src}/Model.cpp -o ${build}/Model.o

Unigram.o: ${src}/Unigram.cpp
	$(CC) $(CFLAGS) ${src}/Unigram.cpp -o ${build}/Unigram.o

Vocabulary.o: ${src}/Vocabulary.cpp
	$(CC) $(CFLAGS) ${src}/Vocabulary.cpp -o ${build}/Vocabulary.o

.PHONY: clean
clean:
	rm -r ./build/*.o ./kernelNodeEmb

$(shell   mkdir -p $(build))