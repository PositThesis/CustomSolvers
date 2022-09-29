#pragma once

#include <Eigen/Dense>
#include <utility>
#include <cmath>

using Eigen::MatrixX;
using Eigen::VectorX;

template <typename Scalar>
struct HouseholderResult {
    VectorX<Scalar> w_n;
    VectorX<Scalar> v_n;
    VectorX<Scalar> h_n_last;
    VectorX<Scalar> z_n;
    Scalar beta;
};

template <typename Scalar>
VectorX<Scalar> apply_householder_vector(VectorX<Scalar> &w, VectorX<Scalar> &rhs) {
  return rhs - 2 * (w * ( w.transpose() * rhs));
}

template <typename Scalar>
std::pair<VectorX<Scalar>, Scalar> make_householder_vector(VectorX<Scalar> &to_rotate, Eigen::Index row) {
  VectorX<Scalar> householder_vector = VectorX<Scalar>::Zero(to_rotate.rows());
  if (row >= to_rotate.rows()) return std::pair(householder_vector, 0);
  householder_vector.tail(to_rotate.rows() - row) = to_rotate.tail(to_rotate.rows() - row);
  Scalar beta = householder_vector.norm();
  if (householder_vector(row) < 0) beta *= -1;
  householder_vector(row) -= beta;
  if (householder_vector.norm() == 0) return std::pair(VectorX<Scalar>::Zero(to_rotate.rows()), 0);
  return std::pair(householder_vector/householder_vector.norm(), beta);
}

template <typename Scalar, typename MatrixType>
HouseholderResult<Scalar> householder_step(MatrixX<Scalar> &W, MatrixType &A, VectorX<Scalar> &z_n_last) {
  auto householder_result = make_householder_vector<Scalar>(z_n_last, W.cols());
  VectorX<Scalar> w_n = householder_result.first;
  Scalar beta = householder_result.second;

  VectorX<Scalar> h_n_last = apply_householder_vector<Scalar>(w_n, z_n_last);

  VectorX<Scalar> v_n = VectorX<Scalar>::Zero(z_n_last.rows());
  if (W.cols() < v_n.rows()) v_n(W.cols()) = 1;
  v_n = apply_householder_vector<Scalar>(w_n, v_n);
  for (Eigen::Index col = W.cols() - 1; col >= 0; col--) {
    VectorX<Scalar> w_col = W.col(col);
    v_n = apply_householder_vector<Scalar>(w_col, v_n);
  }

  VectorX<Scalar> z_n = A * v_n;

  for (Eigen::Index col = 0; col < W.cols(); col++) {
    VectorX<Scalar> w_col = W.col(col);
    z_n = apply_householder_vector<Scalar>(w_col, z_n);
  }
  z_n = apply_householder_vector<Scalar>(w_n, z_n);

  HouseholderResult<Scalar> result;
  result.v_n = v_n;
  result.h_n_last = h_n_last;
  result.w_n = w_n;
  result.z_n = z_n;
  result.beta = beta;
  return result;
}
