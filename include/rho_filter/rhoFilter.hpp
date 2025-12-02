#ifndef RHOFILTERHPP
#define RHOFILTERHPP

#include <Eigen/Dense>

class rhoFilter {
public:
    rhoFilter(double sampling_time,
               int state_space_dim,
               double alpha,
               double k1,
               double k2,
               double k3);

    Eigen::MatrixXd propagate_filter(const Eigen::MatrixXd&);

    // Internal state of the filter
    Eigen::MatrixXd zeta;

private:
    double sampling_time;
    double alpha;
    double k1, k2, k3;

    int zeta_dim, state_space_dim;
    Eigen::MatrixXd m, n, e, s;
    Eigen::MatrixXd I_4, I_n;
    double m_12, m_13, m_14, m_32, m_33, m_42, m_43;
    double n_12, n_13, n_14;
    Eigen::MatrixXd m_inv, a_d, b_q, b_s, A_d, B_q, B_s, S;
    Eigen::MatrixXd v, next_zeta;
};

#endif