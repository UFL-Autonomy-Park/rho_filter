#include "rhoFilter.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct>

Eigen::MatrixXd matrix_signum_elementwise(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd res(x.rows(), x.cols());
    for (int i = 0; i < x.size(); ++i) {
        double element = x(i);
        if (element > 0) res(i) = 1.0;
        else if (element < 0) res(i) = -1.0;
        else res(i) = 0.0;
    }
    return res;
}

rhoFilter::rhoFilter(
    double sampling_time,
    int state_space_dim,
    double alpha,
    double k1,
    double k2,
    double k3
) : 
    sampling_time(sampling_time),
    alpha(alpha),
    k1(k1),
    k2(k2),
    k3(k3),
    state_space_dim(state_space_dim),
    M(4 * state_space_dim, 4 * state_space_dim),
    N(4 * state_space_dim, state_space_dim),
    E(4 * state_space_dim, state_space_dim),
    S(state_space_dim, 4 * state_space_dim),
    I_4n(Eigen::MatrixXd::Identity(4 * state_space_dim, 4 * state_space_dim))
{
    m_12 = -(3 - pow(k1,2) + (2*k1 + k2 + k3)*(k1 + k2));
    m_13 = -(k1 + k2);
    m_14 = -(1 + (k1 + k2)*(k3 - k1));
    m_32 = -((2*k1 + k2 + k3)*(k1 + k2) + 1);
    m_33 = -(2*k1 + k2);
    m_42 = -(2*k1 + k2 + k3);
    m_43 = -(k1 + k2)*(k3 - k1);

    m << 0,     1,      0,      0,
         m_12,  0,      m_32,   m_42,
         m_13,  0,      m_33,  -1,
         m_14,  0,      m_43,  -k3;
    M = Eigen::kroneckerProduct(m, I_4n).eval()

    n_12 = 3 - pow(k1,2) + (2*k1 + k2 + k3)*(k1 + k2);
    n_13 = k1 + k2;
    n_14 = 1 + (k1 + k2)*(k3 - k1);

    n << 0,
         n_12,
         n_13,
         n_14;
    N = Eigen::kroneckerProduct(n, I_4n).eval()    

    e << 0,
         1,
         0,
         0;
    E = Eigen::kroneckerProduct(e, I_4n).eval()

    s << -1, 0, -1, 0;
    S = Eigen::kroneckerProduct(s, I_4n).eval()

    M_inv = M.colPivHouseholderQr().inverse();
    A_d   = (M * sampling_time).exp();
    B_q   = M_inv * (A_d - I_4n) * N;
    B_s   = M_inv * (A_d - I_4n) * E;

    zeta.resize(4 * state_space_dim, 1); // Resize because it's set in .hpp without a size
    zeta.setZero(); // Initialize filter at zero
}

void Rho_Filter::propagate_filter(Eigen::MatrixXd last_position) // q_k, which must be a (n,1)
{
    // S is (n,4n)
    // N is (4n,n)
    // E is (4n,n)
    // M is (4n,4n)
    // zeta is (4n,1)
    // A_d is (4n,4n)
    // B_q is (4n,n)
    // B_s is (4n,n)
    Eigen::MatrixXd v = S * zeta + last_position;
    this->zeta = A_d * this->zeta
         + B_q * last_position
         + alpha * B_s * matrix_signum_elementwise(v)(0,0);
}
