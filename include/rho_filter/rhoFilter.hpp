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

    void propagate_filter(const Eigen::MatrixXd&);

    // Internal state of the filter
    Eigen::MatrixXd zeta;

private:
    double sampling_time;
    double alpha;
    double k1, k2, k3;
    Eigen::MatrixXd M, N, E, S;
    Eigen::MatrixXd M_inv, A_d, I_n, I_4n, B_q, B_s;
    Eigen::MatrixXd m, n, e, s;
    double m_12, m_13, m_14, m_32, m_33, m_42, m_43;
    double n_12, n_13, n_14;
    Eigen::MatrixXd v;
    Eigen::MatrixXd next_zeta;
};

#endif