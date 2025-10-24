# MNIST Digit Classifier (C++)

A demo-friendly, research-oriented MNIST digit classifier implemented in C++ with an interactive frontend using the Canvas API. This project lets users draw digits in the browser and see predictions from a C++ neural network in real-time. Perfect for minor projects, portfolios, or research exploration.

## Project Overview

- Frontend: HTML + JavaScript (Canvas API) for drawing digits.
- Backend: C++ neural network trained on the MNIST dataset.
- Communication: HTTP / WebSocket API bridge between frontend and backend.
- Goal: Interactive, diary-ready, research-heavy demo in a short timeframe.

## Features

- Draw digits on a web canvas and submit to the C++ backend.
- Real-time digit classification with confidence scores.
- Pre-trained MNIST weights for instant demo without retraining.
- Optional visualizations: prediction history, confidence charts.

### Getting Started
#### Prerequisites
- C++ compiler (G++ recommended)
- CMake build system
- Node.js or Live Server for frontend (optional)
- Python (only if using Flask wrapper for C++ backend)

#### Installation
1. Clone the repository:
```
git clone https://github.com/awaisAhmed19/MNIST.git
cd MNIST
```

2. Build C++ backend:
```
mkdir build && cd build
cmake ..
make
```
3. Start frontend server:
```
Option 1: VS Code Live Server
Option 2: Node.js static server
npx serve ../frontend
```
4. Run backend:
```
./mnist_backend  # or python wrapper if applicable

Open browser at: http://localhost:3000 (or server port)
```

### Usage
- Draw a digit (0–9) on the canvas.
- Click Submit to send the digit to the backend.
- View the predicted digit and confidence on the screen.
- Optional: use Clear button to start over.

#### Project Structure
```
```
```
MNIST/
├── backend/                  # C++ neural network backend
│   ├── CMakeLists.txt        # Build instructions for backend
│   ├── main.cpp              # Entry point: training + evaluation
│   ├── NN/                   # Neural network implementation
│   │   ├── neural_network.cpp
│   │   └── neural_network.h
│   ├── Matrix/               # Matrix class implementation
│   │   ├── matrix.cpp
│   │   └── matrix.h
│   ├── Filer.cpp             # File I/O utilities (load/save CSV & descriptors)
│   ├── Filer.h
│   └── utils.h               # Any helper functions (optional)
│
├── frontend/                 # Optional: web-based UI (MERN stack)
│   ├── client/               # React app
│   │   ├── public/
│   │   │   └── index.html
│   │   └── src/
│   │       ├── App.js
│   │       ├── index.js
│   │       └── components/
│   └── server/               # Node/Express server
│       ├── server.js
│       ├── routes/
│       └── models/
│
├── data/                     # MNIST datasets
│   ├── train.csv
│   └── test.csv
│
├── pretrained/               # Pre-trained network weights
│   ├── hidden.csv
│   └── output.csv
│
├── build/                    # CMake build folder (auto-generated)
├── README.md
├── LICENSE
├── .gitignore
└── build.sh                  # Script to build and run backend

```

### Technologies
- C++17/20 – core neural network logic
- HTML5 Canvas API – drawing interface
- JavaScript – sending/receiving data from backend
- CMake – build system
- Optional: Python Flask / Boost.Beast / cpp-httplib for API

### Training Your Own Model
Load MNIST dataset (28×28 grayscale images).
Train a feedforward neural network (MLP) using backpropagation.
Save trained weights (.bin or .txt) into pretrained/.
Backend will load weights on startup for predictions.

### Contributing

- Modular commits encouraged (frontend/backend separation).
- Feature suggestions: prediction history, visualization charts, undo functionality.
- Open to optimizations in training, architecture, or API performance.
