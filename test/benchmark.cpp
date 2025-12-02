#include "rho_filter/rhoFilter.hpp"
#include <iostream>
#include <chrono>

int main() {
    int n = 4;
    rhoFilter f(0.01, n, 2.0, 10.0, 2.0, 2.0);
    
    Eigen::MatrixXd q(n, 1);
    q.setOnes();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < 1000000; ++i) {
        q(0, 0) += 0.0001; 
        f.propagate_filter(q);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
    
    return 0;
}