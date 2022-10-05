#pragma once

#include <Eigen/Dense>

using Eigen::MatrixX;
using Eigen::VectorX;

template <typename Scalar>
struct LanczosLAIterResult {
    bool regular;
    VectorX<Scalar> v_next;
    VectorX<Scalar> w_next;
    VectorX<Scalar> h_n;
};

// k is the 0 based lanczos block index
template <typename Scalar, typename MatrixType>
LanczosLAIterResult<Scalar> lanczos_la_step(
    Eigen::Index n,
    Eigen::Index k,
    MatrixX<Scalar> &V_k,
    MatrixX<Scalar> &W_k,
    MatrixX<Scalar> &V_k_last,
    MatrixX<Scalar> &W_k_last,
    MatrixX<Scalar> &D_last,
    MatrixType &A,
    VectorX<Scalar> &rhs
) {
  MatrixX<Scalar> D = W_k.adjoint() * V_k;
  // singular values of D are the roots of the eigenvalues of D.adjoint() * D
  Eigen::SelfAdjointEigenSolver<MatrixX<Scalar>> es(D.adjoint() * D, false);
  Scalar min_singular = es.eigenvalues().cwiseAbs().minCoeff();
  bool regular = min_singular > 1e-7;

  VectorX<Scalar> alpha = VectorX<Scalar>::Zero(W_k.cols());
  VectorX<Scalar> alphaW = VectorX<Scalar>::Zero(W_k.cols());
  VectorX<Scalar> beta; // = VectorX<Scalar>::Zero(V_k.cols());
  VectorX<Scalar> betaW; // = VectorX<Scalar>::Zero(V_k.cols());

  VectorX<Scalar> v_n = V_k.col(V_k.cols() - 1);
  VectorX<Scalar> w_n = W_k.col(W_k.cols() - 1);

  // only compute these once, are needed for alpha and beta again
  VectorX<Scalar> A_v_n = A * v_n;
  VectorX<Scalar> A_adj_w_n = A.adjoint() * w_n;

  VectorX<Scalar> v_next = A_v_n;
  VectorX<Scalar> w_next = A_adj_w_n;

  VectorX<Scalar> h_n = VectorX<Scalar>::Zero(n+2);

  if (regular) {
    MatrixX<Scalar> D_inv;
    if (D.rows() > 1) {
      D_inv = D.inverse();
    } else {
      D_inv = MatrixX<Scalar>::Zero(1, 1);
      D_inv.coeffRef(0, 0) = 1/D.coeff(0, 0);
    }

    alpha = D_inv * (W_k.adjoint() * A_v_n);
    alphaW = D_inv * (V_k.adjoint() * A_adj_w_n);

    v_next -= V_k * alpha;
    w_next -= W_k * alphaW;
  }

  if (k > 0) {
    MatrixX<Scalar> D_last_inv;
    if (D_last.rows() > 1) {
      D_last_inv = D_last.inverse();
    } else {
      D_last_inv = MatrixX<Scalar>::Zero(1, 1);
      D_last_inv.coeffRef(0, 0) = 1/D_last.coeff(0, 0);
    }

    beta = D_last_inv * (W_k_last.adjoint() * A_v_n);
    betaW = D_last_inv * (V_k_last.adjoint() * A_adj_w_n);
    Eigen::Index leading_zeros = (n + 2) - 1 - alpha.rows() - beta.rows(); // total rows - rho - alpha - beta

    v_next -= V_k_last * beta;
    w_next -= W_k_last * betaW;

    h_n.block(leading_zeros,               0, beta.rows(), 1) = beta;
    h_n.block(leading_zeros + beta.rows(), 0, alpha.rows(),1) = alpha;
  } else {
    h_n.block(0, 0, alpha.rows(), 1) = alpha;
  }

  Scalar rho = v_next.norm();
  Scalar xi = w_next.norm();

  if (rho < 1e-7 || xi == 1e-7)
    throw false; // there is only one throw condition, no need for special types

  v_next /= rho;
  w_next /= xi;

  h_n(n+1) = rho; // h_n has n+2 rows: n+1 is the last row

  LanczosLAIterResult<Scalar> result;
  result.h_n = h_n;
  result.v_next = v_next;
  result.w_next = w_next;
  result.regular = regular;

  return result;
}

