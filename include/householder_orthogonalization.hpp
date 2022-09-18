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
};

template <typename Scalar>
VectorX<Scalar> apply_householder_vector(VectorX<Scalar> &w, VectorX<Scalar> &rhs) {
  return rhs - 2 * (w * ( w.transpose() * rhs));
}

template <typename Scalar>
VectorX<Scalar> make_householder_vector(VectorX<Scalar> &to_rotate, Eigen::Index row) {
  MatrixX<Scalar> sum = to_rotate.tail(to_rotate.rows() - row).transpose() * to_rotate.tail(to_rotate.rows() - row);
  Scalar beta = std::sqrt(sum(0, 0));
  if (to_rotate(row) < 0) beta *= -1;
  VectorX<Scalar> householder_vector = to_rotate;
  if (row > 0) {
    householder_vector.head(row - 1).setZero();
  }
  householder_vector(row) += beta;
  return householder_vector/householder_vector.norm();
}

template <typename Scalar>
HouseholderResult<Scalar> householder_step(MatrixX<Scalar> &W, MatrixX<Scalar> &A, VectorX<Scalar> &z_n_last) {
  VectorX<Scalar> w_n = make_householder_vector(z_n_last, W.cols()); // z_n_last / z_n_last.norm();
  std::cout << "w: " << w_n.transpose() << std::endl;

  std::cout << "Pz: " << apply_householder_vector<Scalar>(w_n, z_n_last).transpose() << std::endl;

  VectorX<Scalar> h_n_last = apply_householder_vector<Scalar>(w_n, z_n_last);
  std::cout << "h: " << h_n_last.transpose() << std::endl;

  VectorX<Scalar> v_n = VectorX<Scalar>::Zero(z_n_last.rows());
  v_n(W.cols()) = 1;

  for (Eigen::Index col = W.cols() - 1; col >= 0; col--) {
    VectorX<Scalar> w_col = W.col(col);
    v_n = apply_householder_vector<Scalar>(w_col, v_n);
  }
  std::cout << "v: " << v_n.transpose() << std::endl;

  VectorX<Scalar> z_n = A * v_n;

  for (Eigen::Index col = 0; col < W.cols(); col++) {
    VectorX<Scalar> w_col = W.col(col);
    z_n = apply_householder_vector<Scalar>(w_col, z_n);
  }
  std::cout << "z: " << z_n.transpose() << std::endl;

  HouseholderResult<Scalar> result;
  result.v_n = v_n;
  result.h_n_last = h_n_last;
  result.w_n = w_n;
  result.z_n = z_n;
  return result;
}
