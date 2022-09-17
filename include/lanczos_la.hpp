#pragma once

#include <Eigen/Dense>
#include <tuple>

using Eigen::MatrixX;
using Eigen::VectorX;

template <typename Scalar>
struct LanczosLAIterResult {
    bool regular;
    VectorX<Scalar> v_next;
    VectorX<Scalar> w_next;
    VectorX<Scalar> h_next;
};

// k is the 1 based lanczos block index
template <typename Scalar>
LanczosLAIterResult<Scalar> lanczos_la_step(
    Eigen::Index k,
    MatrixX<Scalar> &V_k,
    MatrixX<Scalar> &W_k,
    MatrixX<Scalar> &V_k_last,
    MatrixX<Scalar> &W_k_last,
    MatrixX<Scalar> D_last,
    MatrixX<Scalar> A,
    VectorX<MatrixX> rhs
) {
  MatrixX<Scalar> D = W_k.adjoint * V_k;
  Eigen::SelfAdjointEigenSolver<Scalar> es(Decision, false);
  Scalar min_singular = es.eigenvalues().abs().minCoeff();
  bool regular = min_singular < Scalar(1e-7);

  VectorX<Scalar> alpha = VectorX<Scalar>::Zero(V_k.cols());
  VectorX<Scalar> alphaW = VectorX<Scalar>::Zero(V_k.cols());
  VectorX<Scalar> beta = VectorX<Scalar>::Zero(V_k.cols());
  VectorX<Scalar> betaW = VectorX<Scalar>::Zero(V_k.cols());

  VectorX<Scalar> v_n = V_k.col(V_k.cols() - 1);
  VectorX<Scalar> w_n = W_k.col(W_k.cols() - 1);

  VectorX<Scalar> v_next = v_next = A * v_n;
  VectorX<Scalar> w_next = w_next = A.adjoint() * w_n;

  VectorX<Scalar> h_next = VectorX<Scalar>::Zero(k+1);
  Eigen::Index leading_zeros = (k + 1) - 1 - alpha.rows() - beta.rows();

  if (regular) {
    MatrixX<Scalar> D_inv = D.inverse();

    alpha = D_inv * W_k.adjoint() * A * v_n;
    alphaW = D_inv * V_k.adjoint() * A.adjoint() * w_n;

    v_next -= V_k * alpha;
    w_next -= W_k * alphaW;
  }

  if (k > 1) {
    MatrixX<Scalar> D_last_inv = D_last.inverse();
    beta = D_last_inv * W_k.adjoint() * A * v_n;
    betaW = D_last_inv * V_k.adjoint() * A.adjoint() * w_n;

    v_next -= V_k_last * beta;
    w_next -= W_k_last * betaW;

    h_next.block(leading_zeros, beta.rows()) = beta;
    h_next.block(leading_zeros + beta.rows(), alpha.rows()) = alpha;
  } else {
    h_next.block(0, alpha.rows()) = alpha;
  }

  Scalar rho = v_next.norm();
  Scalar xi = w_next.norm();

  if (rho == 0 || xi == 0)
    throw false; // there is only one throw condition, no need for fancy types

  v_next /= rho;
  w_next /= xi;

  h_next(k) = rho; // h_next has k+1 rows: k is the last row

  LanczosLAIterResult<Scalar> result;
  result.h_next = h_next;
  result.v_next = v_next;
  result.w_next = w_next;
  result.regular = regular;
  return result;
}

