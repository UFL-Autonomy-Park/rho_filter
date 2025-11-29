#include <emscripten/bind.h>
#include <vector>
#include "rho_filter/rhoFilter.hpp"

using namespace emscripten;

// Wrapper to handle Eigen <-> std::vector conversions for JS
class RhoFilterWASM {
public:
    RhoFilterWASM(double ts, int dim, double alpha, double k1, double k2, double k3) {
        // Instantiate the actual filter
        filter = std::make_unique<rhoFilter>(ts, dim, alpha, k1, k2, k3);
    }

    // JS passes a simple array (vector), we convert to Eigen, run filter, return array
    std::vector<double> update(const std::vector<double>& input_flat) {
        int dim = input_flat.size();
        
        // Map std::vector to Eigen::Vector
        Eigen::Map<const Eigen::VectorXd> input(input_flat.data(), dim);
        
        // Run logic
        filter->propagate_filter(input);

        // Extract result (zeta is 4*n). 
        // We copy the whole state vector back to JS.
        std::vector<double> output(filter->zeta.size());
        Eigen::VectorXd::Map(output.data(), output.size()) = filter->zeta;
        
        return output;
    }

private:
    std::unique_ptr<rhoFilter> filter;
};

// Expose to JavaScript
EMSCRIPTEN_BINDINGS(rho_filter_module) {
    register_vector<double>("VectorDouble");

    class_<RhoFilterWASM>("RhoFilter")
        .constructor<double, int, double, double, double, double>()
        .function("update", &RhoFilterWASM::update);
}