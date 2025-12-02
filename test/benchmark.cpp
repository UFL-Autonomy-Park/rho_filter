#include "rho_filter/rhoFilter.hpp"
#include <iostream>
#include <chrono>

int main() {
    int n = 4;
    rhoFilter f(0.01, n, 2.0, 10.0, 2.0, 2.0);
    
    // Match Rust initialization: All 1.0
    Eigen::MatrixXd q(n, 1);
    q.setOnes();
    
    double p_sim = 0.0;
    double omega = 2.0;
    double dt = 0.01;

    Eigen::MatrixXd result;

    auto start = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < 1000000; ++i) {
        // Physics Loop (Harmonic Oscillator) on Index 0
        q(0, 0) += p_sim * dt;
        p_sim += -omega * omega * q(0, 0) * dt;
        
        result = f.propagate_filter(q);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
    
    if (result.size() > 0) {
        std::cout << "Checksum: " << result(0, 0) << std::endl;
    }
    
    return 0;
}