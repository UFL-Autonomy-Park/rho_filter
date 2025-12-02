# Rho Filter Wasm Simulation

This project implements a Lur’e Observer using C++ (Eigen) compiled to WebAssembly, with a React frontend.

## Prerequisites

* C++ Compiler (GCC/Clang/MSVC)

* Git

* Python 3 (for local web server)

## 1. Setup & Dependencies

### Windows

1. Install Eigen (Linear Algebra Library):

   * Open PowerShell.

   * Navigate to the project root.

   * Run: git clone https://gitlab.com/libeigen/eigen.git eigen

2. Install Emscripten (Wasm Compiler):

   * Navigate to the project root.

   * Run:
     git clone https://github.com/emscripten-core/emsdk.git
     cd emsdk
     .\emsdk.bat install latest
     .\emsdk.bat activate latest

   * Critical: Run ".\emsdk_env.bat" in your terminal to set environment variables before compiling.

### macOS

1. Install Eigen:

   * Run: brew install eigen

   * (Default path: /opt/homebrew/include/eigen3 or /usr/local/include/eigen3)

2. Install Emscripten:

   * Run: brew install emscripten

### Linux (Ubuntu/Debian)

1. Install Eigen:

   * Run: sudo apt-get install libeigen3-dev

   * (Default path: /usr/include/eigen3)

2. Install Emscripten:

   * Navigate to project root.

   * Run:
     git clone https://github.com/emscripten-core/emsdk.git
     cd emsdk
     ./emsdk install latest
     ./emsdk activate latest
     source ./emsdk_env.sh

## 2. Compilation

Run these commands from the **project root**.

### Windows PowerShell or macOS/Linux

(Adjust include path -I if your Homebrew location differs)

emcc -O3 -Iinclude -I/opt/homebrew/include/eigen3 src/rhoFilter.cpp test/gui_test/bindings.cpp -o test/gui_test/rho_wasm.js -s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME="createRhoModule" -s ALLOW_MEMORY_GROWTH=1 -s NO_DISABLE_EXCEPTION_CATCHING --bind

## 3. Running

Browsers block Wasm loading from local files. Serve via HTTP from the project root:

python3 -m http.server -d test/gui_test 8000

Open http://localhost:8000 in your browser.


# Rust vs C++ Benchmark


## C++ Benchmark 

In the root project directory, run

```
mkdir build
cd build
cmake ..
make
./bench_cpp
```

The CMake is currently optimized for release for a "best case" comparison.

## Rust Benchmark

In the root project directory, run

```
cd test/rust_benchmark
cargo run --release
```