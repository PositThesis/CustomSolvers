#pragma once

#include <Eigen/Dense>
#include <utility>
#include <cmath>
#include <iostream>

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
  rhs.tail(w.size()) -= 2 * (w * ( w.transpose() * rhs.tail(w.size())));
  return rhs;
}

template <typename Scalar>
std::pair<VectorX<Scalar>, Scalar> make_householder_vector(VectorX<Scalar> &to_rotate, Eigen::Index pivot) {
  Eigen::Index vector_size = to_rotate.size() - pivot;
  if (vector_size < 1) return std::pair(VectorX<Scalar>::Zero(0), 0);
  VectorX<Scalar> householder_vector = to_rotate.tail(to_rotate.rows() - pivot);
  Scalar beta = householder_vector.norm();
  if (householder_vector(0) > 0) beta *= -1;
  householder_vector(0) -= beta;
  if (householder_vector.norm() == 0) return std::pair(VectorX<Scalar>::Zero(0), 0);
  return std::pair(householder_vector/householder_vector.norm(), beta);
}

template <typename Scalar, typename MatrixType>
HouseholderResult<Scalar> householder_step(std::vector<VectorX<Scalar>> &W, MatrixType &A, VectorX<Scalar> &z_n_last) {
  auto householder_result = make_householder_vector<Scalar>(z_n_last, W.size());
  VectorX<Scalar> w_n = householder_result.first;
  Scalar beta = householder_result.second;


  VectorX<Scalar> h_n_last = apply_householder_vector<Scalar>(w_n, z_n_last);

  VectorX<Scalar> v_n = VectorX<Scalar>::Zero(z_n_last.rows());
  if (W.size() < v_n.rows()) v_n(W.size()) = 1;
  v_n = apply_householder_vector<Scalar>(w_n, v_n);
  for (Eigen::Index col = W.size() - 1; col >= 0; col--) { // be careful, this must be signed
    VectorX<Scalar> w_col = W[col];
    v_n = apply_householder_vector<Scalar>(w_col, v_n);
  }

  VectorX<Scalar> z_n = A * v_n;

  for (uint64_t col = 0; col < W.size(); col++) {
    VectorX<Scalar> w_col = W[col];
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

template <typename Scalar>
struct HouseholderData {
  VectorX<Scalar> hh_tail;
  Scalar tau;
  Scalar beta;
};

template <typename Scalar>
struct HouseholderStepResult {
  HouseholderData<Scalar> hh_data;
  VectorX<Scalar> h_n;
  VectorX<Scalar> v_n;
  bool last; // if last is set, both the hh_data as well as h_n are invalid
};

template <typename Scalar>
HouseholderData<Scalar> make_householder_vector_like_eigen(Eigen::Ref<VectorX<Scalar>> input) {
  Scalar beta = (input.coeffRef(0) > 0 ? -1 : 1) * input.norm();
  VectorX<Scalar> hh = input.tail(input.size() - 1) / (input.coeffRef(0) - beta);
  Scalar tau = (beta - input.coeffRef(0)) / beta;

  HouseholderData<Scalar> data;
  data.hh_tail = hh;
  data.beta = beta;
  data.tau = tau;

  return data;
}

template <typename Scalar>
void apply_householder_like_eigen(Eigen::Ref<VectorX<Scalar>> input, HouseholderData<Scalar> &hh_data) {
  Eigen::Index pivot = input.size() - (hh_data.hh_tail.size() + 1);
  MatrixX<Scalar> tmp_1 = hh_data.hh_tail.adjoint() * input.tail(hh_data.hh_tail.size());
  Scalar tmp = tmp_1.coeffRef(0, 0) + input.coeffRef(pivot);
  input.coeffRef(pivot) -= hh_data.tau * tmp;
  input.tail(hh_data.hh_tail.size()) -= hh_data.tau * hh_data.hh_tail * tmp;
}

template <typename Scalar, typename MatrixType>
HouseholderStepResult<Scalar> householder_step_like_eigen(std::vector<HouseholderData<Scalar>> &W, MatrixType &A) {
  Eigen::Index k = W.size();
  Eigen::Index m = A.rows();

  if (m < k) {
    // cannot make such a unit vector
    throw false;
  }

  VectorX<Scalar> v = VectorX<Scalar>::Unit(m, k-1);

  for (int32_t idx = W.size() - 1; idx >= 0; idx--) {
    apply_householder_like_eigen<Scalar>(v, W[idx]);
  }

  // store the column v, it's the new basis vector
  Eigen::VectorX<Scalar> v_n = v;

  v = A * v;

  for (uint32_t idx = 0; idx < W.size(); idx++) {
    apply_householder_like_eigen<Scalar>(v, W[idx]);
  }

  HouseholderStepResult<Scalar> result;
  // first the normal case
  if (m != k) {
    VectorX<Scalar> v_copy = v;
    HouseholderData<Scalar> hh_data = make_householder_vector_like_eigen<Scalar>(v.tail(m - k));
    apply_householder_like_eigen<Scalar>(v, hh_data); // turn v into h_n

    VectorX<Scalar> eigen_hh = VectorX<Scalar>::Zero(m-k);
    VectorX<Scalar> eigen_scratch = VectorX<Scalar>::Zero(m-k);

    Scalar beta, tau;
    v_copy.tail(m-k).makeHouseholder(eigen_hh, tau, beta);
    v_copy.tail(m-k).applyHouseholderOnTheLeft(eigen_hh, tau, eigen_scratch.data());

    result.hh_data = hh_data;
    result.h_n = v; // be careful with the names here.
    result.last = false;
  } else {
    // then the special case:
    // here m == k (if it were more, we were in the if-branch, if it were less, we would have thrown an exception)
    result.last = true;
  }
  result.v_n = v_n;

  return result;
}
