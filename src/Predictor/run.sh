
#!/bin/bash

echo "[hot] watching for changes..."

ls ../NN/*.cpp ../Tensor/*.cpp ../Filer.cpp *.cpp *.h | entr -r bash -c '
  clear
  ./build.sh && ./app
'
