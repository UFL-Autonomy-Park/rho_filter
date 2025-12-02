use nalgebra::{DMatrix, DVector};
use std::time::Instant;

struct RhoFilter {
    alpha: f64,
    // Pre-computed discrete matrices
    ad: DMatrix<f64>,
    bq: DMatrix<f64>,
    bs: DMatrix<f64>,
    s: DMatrix<f64>,
    // State vectors
    zeta: DVector<f64>,
    v: DVector<f64>,
    next_zeta: DVector<f64>, 
}

impl RhoFilter {
    fn new(dt: f64, n: usize, alpha: f64, k1: f64, k2: f64, k3: f64) -> Self {
        // --- Matrix Setup (Identical Logic) ---
        let m12 = -(3.0 - k1.powi(2) + (2.0*k1 + k2 + k3)*(k1 + k2));
        let m13 = -(k1 + k2);
        let m14 = -(1.0 + (k1 + k2)*(k3 - k1));
        let m32 = -((2.0*k1 + k2 + k3)*(k1 + k2) + 1.0);
        let m33 = -(2.0*k1 + k2);
        let m42 = -(2.0*k1 + k2 + k3);
        let m43 = -(k1 + k2)*(k3 - k1);

        let m = DMatrix::from_row_slice(4, 4, &[
            0.0, 1.0, 0.0, 0.0,
            m12, 0.0, m32, m42,
            m13, 0.0, m33, -1.0,
            m14, 0.0, m43, -k3,
        ]);

        let det = k1.powi(4) + 2.0*k1.powi(3)*k2 + k1.powi(2)*k2.powi(2) 
                + k1.powi(2)*k2*k3 + k1*k2 + 2.0*k1*k3 + 1.0;
        let idet = 1.0 / det;

        let mut m_inv = DMatrix::zeros(4, 4);
        m_inv[(0, 1)] = (k1*(-k1 - k2 - k3)) * idet;
        m_inv[(0, 2)] = (2.0*k1.powi(3) + 3.0*k1.powi(2)*k2 + k1.powi(2)*k3 + k1*k2.powi(2) + k1*k2*k3 + k3) * idet;
        m_inv[(0, 3)] = (2.0*k1.powi(2) + k1*k2 + k1*k3 - 1.0) * idet;
        m_inv[(1, 0)] = 1.0; 
        m_inv[(2, 1)] = (k1.powi(2) + k1*k2 - 1.0) * idet;
        m_inv[(2, 2)] = (-2.0*k1.powi(3) - 3.0*k1.powi(2)*k2 - k1*k2.powi(2) - k1*k2*k3 + 2.0*k1 + k2 - 2.0*k3) * idet;
        m_inv[(2, 3)] = (3.0 - k1.powi(2)) * idet;
        m_inv[(3, 1)] = (-k1.powi(3) - k1.powi(2)*k2 + k1.powi(2)*k3 + k1*k2*k3 + 2.0*k1 + k2) * idet;
        m_inv[(3, 2)] = (k1.powi(4) + k1.powi(3)*k2 - k1.powi(3)*k3 - k1.powi(2)*k2*k3 - 4.0*k1.powi(2) - 5.0*k1*k2 + k1*k3 - k2.powi(2) + k2*k3 - 1.0) * idet;
        m_inv[(3, 3)] = (-2.0*k1.powi(2)*k2 - k1.powi(2)*k3 - k1*k2.powi(2) - k1*k2*k3 - 5.0*k1 - 2.0*k2) * idet;

        let n12 = 3.0 - k1.powi(2) + (2.0*k1 + k2 + k3)*(k1 + k2);
        let n13 = k1 + k2;
        let n14 = 1.0 + (k1 + k2)*(k3 - k1);

        let n_vec = DMatrix::from_row_slice(4, 1, &[0.0, n12, n13, n14]);
        let e_vec = DMatrix::from_row_slice(4, 1, &[0.0, 1.0, 0.0, 0.0]);
        let s_vec = DMatrix::from_row_slice(1, 4, &[-1.0, 0.0, -1.0, 0.0]);

        let ad_small = (&m * dt).exp();
        let i_4 = DMatrix::<f64>::identity(4, 4);
        let term = &ad_small - &i_4;
        let bq_small = &m_inv * &term * &n_vec;
        let bs_small = &m_inv * &term * &e_vec;

        let i_n = DMatrix::<f64>::identity(n, n);
        
        RhoFilter {
            alpha,
            ad: ad_small.kronecker(&i_n),
            bq: bq_small.kronecker(&i_n),
            bs: bs_small.kronecker(&i_n),
            s: s_vec.kronecker(&i_n),
            zeta: DVector::zeros(4 * n),
            v: DVector::zeros(n),
            next_zeta: DVector::zeros(4 * n),
        }
    }

    #[inline(always)]
    fn update(&mut self, q: &DVector<f64>) {
        // v = S * zeta
        // Optimized: direct write to self.v, no allocation
        self.s.mul_to(&self.zeta, &mut self.v);
        self.v += q; // Vector addition is efficient

        // next_zeta = Ad * zeta
        // Optimized: overwrites next_zeta
        self.ad.mul_to(&self.zeta, &mut self.next_zeta);
        
        // next_zeta += Bq * q
        // Optimized: gemv(alpha, A, x, beta) -> y = alpha*A*x + beta*y
        // Equivalent to Eigen's noalias() accumulation
        self.next_zeta.gemv(1.0, &self.bq, q, 1.0);

        // next_zeta += alpha * Bs * sgn(v)
        self.v.apply(|x| *x = x.signum()); 
        
        // Optimized: Accumulate directly into next_zeta
        self.next_zeta.gemv(self.alpha, &self.bs, &self.v, 1.0);

        self.zeta.copy_from(&self.next_zeta);
    }
}

fn main() {
    let mut f = RhoFilter::new(0.01, 1, 2.0, 10.0, 2.0, 2.0);
    let mut q = DVector::from_element(1, 1.0);
    let mut p = 0.0;
    let omega = 2.0;
    let dt = 0.01;

    let start = Instant::now();
    for _ in 0..1_000_000 {
        q[0] += p * dt;
        p += -omega * omega * q[0] * dt;
        f.update(&q);
    }
    let duration = start.elapsed();

    println!("Time: {}ms", duration.as_millis());
    println!("Checksum: {}", f.zeta[0]);
}