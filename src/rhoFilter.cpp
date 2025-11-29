#include "rho_filter/rhoFilter.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct>
#include <cmath>

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
    k3(k3)
{   
    int zeta_dim = 4 * state_space_dim;
    int num_inputs = state_space_dim;

    m.resize(4, 4);
    n.resize(4, 1);
    e.resize(4, 1);
    s.resize(1, 4);
    M.resize(zeta_dim, zeta_dim);
    N.resize(zeta_dim, num_inputs);
    E.resize(zeta_dim, num_inputs);
    S.resize(num_inputs, zeta_dim);
    I_n = Eigen::MatrixXd::Identity(num_inputs, num_inputs);
    I_4n = Eigen::MatrixXd::Identity(zeta_dim, zeta_dim);

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

    n_12 = 3 - pow(k1,2) + (2*k1 + k2 + k3)*(k1 + k2);
    n_13 = k1 + k2;
    n_14 = 1 + (k1 + k2)*(k3 - k1);

    n << 0,
         n_12,
         n_13,
         n_14;
    e << 0,
         1,
         0,
         0;
    s << -1, 0, -1, 0;

    M = Eigen::kroneckerProduct(m, I_n).eval();
    N = Eigen::kroneckerProduct(n, I_n).eval(); 
    E = Eigen::kroneckerProduct(e, I_n).eval();
    S = Eigen::kroneckerProduct(s, I_n).eval();

    M_inv = M.colPivHouseholderQr().inverse();
    A_d   = (M * sampling_time).exp();
    B_q   = M_inv * (A_d - I_4n) * N;
    B_s   = M_inv * (A_d - I_4n) * E;

    v.resize(num_inputs, 1);
    v.setZero();
    zeta.resize(zeta_dim, 1);
    zeta.setZero(); // Do I initialize it at zero?
    next_zeta.resize(zeta_dim, 1);
    next_zeta.setZero();
}
    
//     // H is (4n + 2n) x (4n + 2n)
//     Eigen::MatrixXd H = Eigen::MatrixXd::Zero(zeta_dim + 2 * num_inputs, zeta_dim + 2 * num_inputs);
    
//     H.topLeftCorner(zeta_dim, zeta_dim) = M;
//     H.block(0, zeta_dim, zeta_dim, num_inputs) = N;            // Place N to the right of M
//     H.block(0, zeta_dim + num_inputs, zeta_dim, num_inputs) = E;   // Place E to the right of N

//     // Exponentiate the composite matrix
//     Eigen::MatrixXd H_exp = (H * sampling_time).exp();

//     // Extract discrete matrices
//     A_d = H_exp.topLeftCorner(zeta_dim, zeta_dim);
//     B_q = H_exp.block(0, zeta_dim, zeta_dim, num_inputs);
//     B_s = H_exp.block(0, zeta_dim + num_inputs, zeta_dim, num_inputs);

//     // Filter state initialization
//     v.resize(num_inputs, 1);
//     v.setZero();
//     zeta.resize(zeta_dim, 1);
//     zeta.setZero(); // Yes, initialize states to zero
//     next_zeta.resize(zeta_dim, 1);
// }

void rhoFilter::propagate_filter(const Eigen::MatrixXd& last_position) // q_k, which must be a (n,1)
{
    // S is (n,4n)
    // N is (4n,n)
    // E is (4n,n)
    // M is (4n,4n)
    // zeta is (4n,1)
    // A_d is (4n,4n)
    // B_q is (4n,n)
    // B_s is (4n,n)

    v.noalias() = S * zeta;
    v += last_position;
    next_zeta.noalias() = A_d * zeta;
    next_zeta.noalias() += B_q * last_position;
    next_zeta.noalias() += alpha * B_s * v.cwiseSign();
    zeta = next_zeta;
}
