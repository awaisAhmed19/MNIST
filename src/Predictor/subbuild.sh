
g++ ./Predictor.cpp ../NN/neural_network.cpp ../Tensor/tensor.cpp ../Filer.cpp DrawWin.c \
-Iinclude \
-lraylib -lm -lpthread -ldl -lrt -lX11 \
-o app

