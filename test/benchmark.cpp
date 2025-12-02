#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class RhoFilter {
public:
    double dt, alpha, k1, k2, k3;
    int n;
    MatrixXd m, m_inv, n_vec, e_vec, s_vec;
    MatrixXd M, N, E, S;
    MatrixXd Ad, Bq, Bs;
    MatrixXd v, zeta, next_zeta;
    MatrixXd In, I4;

    RhoFilter(double _dt, int _n, double _a, double _k1, double _k2, double _k3) 
        : dt(_dt), n(_n), alpha(_a), k1(_k1), k2(_k2), k3(_k3) 
    {
        m.resize(4, 4); m_inv.resize(4, 4);
        n_vec.resize(4, 1); e_vec.resize(4, 1); s_vec.resize(1, 4);
        
        In = MatrixXd::Identity(n, n);
        I4 = MatrixXd::Identity(4, 4);

        double m12 = -(3 - pow(k1,2) + (2*k1 + k2 + k3)*(k1 + k2));
        double m13 = -(k1 + k2);
        double m14 = -(1 + (k1 + k2)*(k3 - k1));
        double m32 = -((2*k1 + k2 + k3)*(k1 + k2) + 1);
        double m33 = -(2*k1 + k2);
        double m42 = -(2*k1 + k2 + k3);
        double m43 = -(k1 + k2)*(k3 - k1);

        m << 0, 1, 0, 0,
             m12, 0, m32, m42,
             m13, 0, m33, -1,
             m14, 0, m43, -k3;

        double det = pow(k1, 4) + 2*pow(k1, 3)*k2 + pow(k1, 2)*pow(k2, 2) + pow(k1, 2)*k2*k3 + k1*k2 + 2*k1*k3 + 1;
        double idet = 1.0 / det;

        m_inv.setZero();
        m_inv(0, 1) = (k1*(-k1 - k2 - k3)) * idet;
        m_inv(0, 2) = (2*pow(k1, 3) + 3*pow(k1, 2)*k2 + pow(k1, 2)*k3 + k1*pow(k2, 2) + k1*k2*k3 + k3) * idet;
        m_inv(0, 3) = (2*pow(k1, 2) + k1*k2 + k1*k3 - 1) * idet;
        m_inv(1, 0) = det * idet;
        m_inv(2, 1) = (pow(k1, 2) + k1*k2 - 1) * idet;
        m_inv(2, 2) = (-2*pow(k1, 3) - 3*pow(k1, 2)*k2 - k1*pow(k2, 2) - k1*k2*k3 + 2*k1 + k2 - 2*k3) * idet;
        m_inv(2, 3) = (3 - pow(k1, 2)) * idet;
        m_inv(3, 1) = (-pow(k1, 3) - pow(k1, 2)*k2 + pow(k1, 2)*k3 + k1*k2*k3 + 2*k1 + k2) * idet;
        m_inv(3, 2) = (pow(k1, 4) + pow(k1, 3)*k2 - pow(k1, 3)*k3 - pow(k1, 2)*k2*k3 - 4*pow(k1, 2) - 5*k1*k2 + k1*k3 - pow(k2, 2) + k2*k3 - 1) * idet;
        m_inv(3, 3) = (-2*pow(k1, 2)*k2 - pow(k1, 2)*k3 - k1*pow(k2, 2) - k1*k2*k3 - 5*k1 - 2*k2) * idet;

        double n12 = 3 - pow(k1,2) + (2*k1 + k2 + k3)*(k1 + k2);
        double n13 = k1 + k2;
        double n14 = 1 + (k1 + k2)*(k3 - k1);

        n_vec << 0, n12, n13, n14;
        e_vec << 0, 1, 0, 0;
        s_vec << -1, 0, -1, 0;

        MatrixXd ad_small = (m * dt).exp();
        MatrixXd bq_small = m_inv * (ad_small - I4) * n_vec;
        MatrixXd bs_small = m_inv * (ad_small - I4) * e_vec;

        Ad = Eigen::kroneckerProduct(ad_small, In);
        Bq = Eigen::kroneckerProduct(bq_small, In);
        Bs = Eigen::kroneckerProduct(bs_small, In);
        S  = Eigen::kroneckerProduct(s_vec, In);

        v.resize(n, 1); v.setZero();
        zeta.resize(4*n, 1); zeta.setZero();
        next_zeta.resize(4*n, 1); next_zeta.setZero();
    }

    void update(const MatrixXd& q) {
        v.noalias() = S * zeta;
        v += q;
        next_zeta.noalias() = Ad * zeta;
        next_zeta.noalias() += Bq * q;
        next_zeta.noalias() += alpha * Bs * v.cwiseSign();
        zeta = next_zeta;
    }
};

int main() {
    RhoFilter f(0.01, 1, 2.0, 10.0, 2.0, 2.0);
    MatrixXd q(1, 1);
    q(0,0) = 1.0;
    
    double p = 0;
    double omega = 2.0;
    double dt = 0.01;

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000000; ++i) {
        q(0,0) += p * dt;
        p += -omega*omega * q(0,0) * dt;
        f.update(q);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() 
              << "ms" << std::endl;
    std::cout << "Checksum: " << f.zeta(0,0) << std::endl;
    return 0;
}