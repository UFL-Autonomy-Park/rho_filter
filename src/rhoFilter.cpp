#include "rho_filter/rhoFilter.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct>
#include <cmath>
#include <iostream>

rhoFilter::rhoFilter(
    double sampling_time,
    int state_space_dim,
    double alpha,
    double k1,
    double k2,
    double k3
) : 
    sampling_time(sampling_time),
    state_space_dim(state_space_dim),
    alpha(alpha),
    k1(k1),
    k2(k2),
    k3(k3)
{   
    n.resize(4, 1);
    e.resize(4, 1);
    s.resize(1, 4);
    m.resize(4, 4);
    m_inv.resize(4, 4);
    zeta_dim = 4 * state_space_dim;

    I_n = Eigen::MatrixXd::Identity(state_space_dim, state_space_dim);
    I_4 = Eigen::MatrixXd::Identity(4, 4);

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

    // The below replaces one line: m_inv = m.colPivHouseholderQr().inverse();
    double det = pow(k1, 4) + 2*pow(k1, 3)*k2 + pow(k1, 2)*pow(k2, 2) + pow(k1, 2)*k2*k3 + k1*k2 + 2*k1*k3 + 1;
    double invDet = 1.0 / det;
    m_inv(0, 0) = 0;
    m_inv(0, 1) = (k1*(-k1 - k2 - k3)) * invDet;
    m_inv(0, 2) = (2*pow(k1, 3) + 3*pow(k1, 2)*k2 + pow(k1, 2)*k3 + k1*pow(k2, 2) + k1*k2*k3 + k3) * invDet;
    m_inv(0, 3) = (2*pow(k1, 2) + k1*k2 + k1*k3 - 1) * invDet;

    m_inv(1, 0) = (pow(k1, 4) + 2*pow(k1, 3)*k2 + pow(k1, 2)*pow(k2, 2) + pow(k1, 2)*k2*k3 + k1*k2 + 2*k1*k3 + 1) * invDet;
    m_inv(1, 1) = 0;
    m_inv(1, 2) = 0;
    m_inv(1, 3) = 0;

    m_inv(2, 0) = 0;
    m_inv(2, 1) = (pow(k1, 2) + k1*k2 - 1) * invDet;
    m_inv(2, 2) = (-2*pow(k1, 3) - 3*pow(k1, 2)*k2 - k1*pow(k2, 2) - k1*k2*k3 + 2*k1 + k2 - 2*k3) * invDet;
    m_inv(2, 3) = (3 - pow(k1, 2)) * invDet;

    m_inv(3, 0) = 0;
    m_inv(3, 1) = (-pow(k1, 3) - pow(k1, 2)*k2 + pow(k1, 2)*k3 + k1*k2*k3 + 2*k1 + k2) * invDet;
    m_inv(3, 2) = (pow(k1, 4) + pow(k1, 3)*k2 - pow(k1, 3)*k3 - pow(k1, 2)*k2*k3 - 4*pow(k1, 2) - 5*k1*k2 + k1*k3 - pow(k2, 2) + k2*k3 - 1) * invDet;
    m_inv(3, 3) = (-2*pow(k1, 2)*k2 - pow(k1, 2)*k3 - k1*pow(k2, 2) - k1*k2*k3 - 5*k1 - 2*k2) * invDet;

    // // Check M * M_inv ~ Identity
    // Eigen::Matrix4d identity_check = m * m_inv;

    // std::cout << "M * M_inv (Should be Identity):" << std::endl;
    // std::cout << identity_check << std::endl;

    // // Quantitative check
    // double error = (identity_check - Eigen::Matrix4d::Identity()).norm();
    // std::cout << "Inverse Error Norm: " << error << std::endl;

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

    a_d = (m * sampling_time).exp();
    b_q = m_inv * (a_d - I_4) * n;
    b_s = m_inv * (a_d - I_4) * e;

    A_d = Eigen::kroneckerProduct(a_d, I_n);
    B_q = Eigen::kroneckerProduct(b_q, I_n);
    B_s = Eigen::kroneckerProduct(b_s, I_n);
    S = Eigen::kroneckerProduct(s, I_n);

    v.resize(state_space_dim, 1);
    v.setZero();
    zeta.resize(zeta_dim, 1);
    zeta.setZero();
    next_zeta.resize(zeta_dim, 1);
    next_zeta.setZero();
}

Eigen::MatrixXd rhoFilter::propagate_filter(const Eigen::MatrixXd& last_position) 
{
    // last_position (q_k) is (n,1)
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

    return zeta.block(state_space_dim, 0, state_space_dim, 1);
}
