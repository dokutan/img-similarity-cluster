# img-similarity-cluster
Find duplicate or similar images

# Installing and running
- Install OpenCV
- Compile with:
```
g++ img-similarity-cluster.cpp -o img-similarity-cluster -std=c++17 -Wall -pthread `pkg-config --cflags --libs opencv4` -O3
```
- Get usage information:
```
./img-similarity-cluster -h
```
