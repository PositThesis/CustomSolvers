#pragma once

#include <Eigen/Dense>
#include <tuple>

using Eigen::MatrixX;
using Eigen::VectorX;

template <typename Scalar>
struct LanczosWLAIterResult {
    VectorX<Scalar> v_next;
    VectorX<Scalar> w_next;
    Scalar rho_next;
    Scalar xi_next;
    VectorX<Scalar> h_next;
    Scalar eps_n;
    Scalar beta_n;
    VectorX<Scalar> p_n;
    VectorX<Scalar> q_n;
}

// n is the 1 based iteration number
template <typename Scalar>
LanczosWLAIterResult<Scalar> lanczos_wla_step(
    Eigen::Index n,
    VectorX<Scalar> &v_n,
    VectorX<Scalar> &w_n,
    VectorX<Scalar> &p_n_last,
    VectorX<Scalar> &q_n_last,
    Scalar xi_n,
    Scalar rho_n,
    Scalar eps_last,
    Scalar beta_n_last,
    MatrixX<Scalar> A,
    VectorX<MatrixX> rhs
) {
    Scalar delta_n = v_n.transpose() * w_n;

    if (delta_n == 0) throw false;

    VectorX<Scalar> p_n = v_n;
    VectorX<Scalar> q_n = w_n;

    if (n > 1) {
        p_n -= p_n_last * xi_n  * delta_n / eps_last;
        q_n -= q_n_last * rho_n * delta_n / eps_last;
    }

    Scalar eps_n = q_n.adjoint() * A * p_n;
    Scalar beta_n = eps_n / delta_n;

    VectorX<Scalar> v_next = A * p_n - v_n * beta_n;
    VectorX<Scalar> w_next = A.adjoint() * q_n - w_n * beta_n;

    Scalar rho_next = v_next.norm();
    Scalar xi_next = w_next.norm();

    if (rho_next == 0 || xi_next == 0) throw false;

    v_next /= rho_next();
    w_next /= xi_next();

    VectorX<Scalar> h_next = VectorX<Scalar>::Zero(n+1);
    if (n == 1) {
        // initial case. Since n is known to be 1, there is only two entries
        h_next(0) = beta_n;
        h_next(1) = rho_next;
    } else {
        Scalar a_current = xi_n * delta_n / eps_last;
        h_next(n - 2) = beta_n_last * a_current;
        h_next(n - 1) = beta_n + a_current * rho_n;
        h_next(n    ) = rho_next;
    }

    LanczosWLAIterResult<Scalar> result;
    result.beta_n = beta_n;
    result.eps_n = eps_n;
    result.h_next = h_next;
    result.p_n = p_n;
    result.q_n = q_n;
    result.rho_next = rho_next;
    result.xi_next = xi_next;
    result.v_next = v_next;
    result.w_next = w_next;
    return result;
}


