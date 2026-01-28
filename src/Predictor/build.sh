
#!/bin/bash
set -e

g++ ../NN/*.cpp ../Tensor/*.cpp ../Filer.cpp *.cpp *.h -o app \
    -std=c++17 \
    -Wall -Wextra \
    -lraylib -lm -lpthread -ldl -lrt -lX11
echo "[build] done"
