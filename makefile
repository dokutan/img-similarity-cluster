CC=g++

build: img-similarity-cluster img-search

img-similarity-cluster:
	$(CC) img-similarity-cluster.cpp -o img-similarity-cluster -std=c++17 -Wall -pthread `pkg-config --cflags --libs opencv4` -O3

img-search:
	$(CC) img-search.cpp -o img-search -std=c++17 -Wall -pthread `pkg-config --cflags --libs opencv4` -O3

install:
	install img-similarity-cluster /usr/bin
	install img-search /usr/bin

uninstall:
	rm /usr/bin/img-similarity-cluster
	rm /usr/bin/img-search

clean:
	rm img-similarity-cluster
	rm img-search
