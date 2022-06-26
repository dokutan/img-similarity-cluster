# img-similarity-cluster
Find duplicate or similar images.

## Installing and running
- Install OpenCV
- Compile and install with:
```
make -j
sudo make install
```

- Get usage information:
```
img-similarity-cluster -h
img-search -h
```

- Search for similar images:
```
img-similarity-cluster -d /path/to/directory
```

- Show similar images in a GUI:
```
img-similarity-cluster -l -d /path/to/directory | view-similar
```
Use the left, right and down arrows to change the image, space prints the current image.

## Comparison with similar tools

6600 images (3.1 GB) on tmpfs:

- img-similarity-cluster : 10 s
- czkawka 4.0.0 : 40 s
- dupeGuru 4.2.1 : 243 s

