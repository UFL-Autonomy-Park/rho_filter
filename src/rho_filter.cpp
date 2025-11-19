#include <iostream>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>



using namespace std;


template <typename T>
int signum(T x){
    return (x < 0) ? -1 : (x > 0);
}

template <typename T>
T matrix_signum_elementwise(const T& A) {
    return A.unaryExpr([](double x) {
        return (double)signum(x); 
    });
}



Rho_Filter::Rho_Filter(
    double sampling_time, // seconds
    int state_space_dim, // State space dim
    double alpha,
    double k1,
    double k2,
    double k3)
   : sampling_time(sampling_time),
      alpha(alpha),
      k1(k1),
      k2(k2),
      k3(k3),

      M(4,4),
      N(4,1),
      E(4,1),
      S(1,4),
{
    int dim = n;
    double m_12 = -(3 - pow(k1, 2) + (2*k1 + k2 + k3) * (k1 + k2));
    double m_13 = -(k1 + k2);
    double m_14 = -(1 + (k1 + k2)*(k3 - k1));
    double m_32 = -((2 * k1 + k2 + k3)*(k1 + k2) + 1);
    double m_33 = -(2 * k1 + k2);
    double m_42 = -(2 * k1 + k2 + k3);
    double m_43 = -(k1 + k2)*(k3 - k1);

    Eigen::MatrixXd M(dim,dim);
    M << 0,    1,   0 ,   0,
        m_12,  0,  m_32,  m_42,
        m_13,  0,  m_33,  -1,
        m_14,  0,  m_43,  -k3;

    double n_12 = 3 - pow(k1, 2) + (2*k1 + k2 + k3)*(k1 + k2);
    double n_13 = k1 + k2;
    double n_14 = 1 + (k1 + k2)*(k3 - k1);

    Eigen::MatrixXd N(dim,1);
    N << 0,
        n_12,
        n_13,
        n_14;
        
    Eigen::MatrixXd E(dim,1);
    E << 0,
        1,
        0,
        0;
    
    Eigen::MatrixXd S(1,dim);
    S << -1, 0, -1, 0;

    // Filter matrices
    Eigen::MatrixXd M_inv = M.colPivHouseholderQr().inverse();
    Eigen::MatrixXd A_d = (M * sampling_time).exp();
    Eigen::MatrixXd I_n = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd B_q = M_inv * (A_d - I_n) * N;
    Eigen::MatrixXd B_s = M_inv * (A_d - I_n) * E;

    // Initial filter state
    Eigen::MatrixXd zeta(4*n,1) // zeta = [ q_hat, p_hat, m, sigma]
}

void Rho_Filter::propagate_filter(double last_position){ // q_k
    zeta = A_d * zeta + B_q * last_position + alpha * B_s * matrix_signum_elementwise(S * zeta + last_position)
}
