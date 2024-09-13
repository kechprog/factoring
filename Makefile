all: main
	./main

main: main.cu
	nvcc -lgmp main.cu -o main
