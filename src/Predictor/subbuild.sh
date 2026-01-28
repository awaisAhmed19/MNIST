g++ -g DrawWin.c nn.c Predictor.cpp ../NN/neural_network.cpp ../Tensor/tensor.cpp ../Filer.cpp \
    -Iinclude \
    -Wall -Wextra -Wshadow -Wconversion -Wsign-conversion \
    -Wformat=2 -Wundef -Wpointer-arith -Wcast-align \
    -Wwrite-strings -Wmissing-declarations \
    -lraylib -lm -lpthread -ldl -lrt -lX11 \
    -o app
