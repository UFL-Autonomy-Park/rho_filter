#include <iostream>
#include <Eigen/Dense>

#include "rho_filter/rhoFilter.hpp"


int main() {
    double sampling_time = 0.001;
    int state_space_dim = 1;
    double alpha = 1.0;
    double k1 = 1.0;
    double k2 = 1.0;
    double k3 = 1.0;

    // Instantiate the filter
    rhoFilter my_filter(sampling_time, state_space_dim, alpha, k1, k2, k3);

    double initial_time = 0.0;
    double t = initial_time;
    Eigen::MatrixXd input_position(state_space_dim, 1);

    while(true) {
        double x = 3.0 * t;
        input_position.setConstant(x);
        my_filter.propagate_filter(input_position);
        std::cout << my_filter.zeta.bottomRows(state_space_dim) << std::endl;

        t += sampling_time;
        
        if (t > 1.0) break;
    }

    return 0;
}