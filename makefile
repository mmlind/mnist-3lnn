all: main

main: 
	gcc -o bin/mnist-3lnn -Iutil -lm main.c 3lnn.c util/screen.c util/mnist-utils.c util/mnist-stats.c

