#ifndef RHOFILTERHPP
#define RHOFILTERHPP

#include <Eigen/Dense>

class Rho_Filter {
public:
    Rho_Filter(double sampling_time,
               int state_space_dim,
               double alpha,
               double k1,
               double k2,
               double k3);

    void propagate_filter(double last_position);

    // Internal state of the filter
    Eigen::MatrixXd zeta;

private:
    double sampling_time;
    double alpha;
    double k1, k2, k3;
    int state_space_dim;

    Eigen::MatrixXd M, N, E, S;
    Eigen::MatrixXd M_inv, A_d, I_n, B_q, B_s;
    double m_12, m_13, m_14, m_32, m_33, m_42, m_43;
    double n_12, n_13, n_14;
};

Eigen::MatrixXd matrix_signum_elementwise(const Eigen::MatrixXd& x);

#endif