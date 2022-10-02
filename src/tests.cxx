#include <Eigen/Dense>
#include "householder_orthogonalization.hpp"
#include <cmath>
#include <iostream>

using Eigen::VectorXd;

bool eigen_householder(VectorXd a) {
    std::cout << a << std::endl;

    double tau, beta;
    VectorXd hh = VectorXd::Zero(3);

    a.makeHouseholder(hh, tau, beta);

    std::cout << "hh: " << hh.transpose() << std::endl;
    std::cout << "beta: " << beta << std::endl;
    std::cout << "tau: " << tau << std::endl;

    std::cout << "tmp: " << hh.adjoint() * a.tail(3) << std::endl;
    std::cout << "tmp: " << a(0) + hh.adjoint() * a.tail(3) << std::endl;

    std::cout << "first: " << a(0) - tau * (a(0) + hh.adjoint() * a.tail(3)) << std::endl;
    std::cout << "rest: " << a.tail(3) - tau * hh * (a(0) + hh.adjoint() * a.tail(3)) << std::endl;

    VectorXd scratch = VectorXd::Zero(4);

    a.applyHouseholderOnTheLeft(hh, tau, scratch.data());

    std::cout << a << std::endl;

    if (std::abs(std::abs(a(0)) - std::sqrt(30)) > 0.1) {
        std::cout << "first element is not sqrt(30)" << std::endl;
        return false;
    }

    for (Eigen::Index idx = 1; idx < a.size(); idx++) {
        if (std::abs(a(idx)) > 0.1) {
            std::cout << idx << ". element is not sqrt(30)" << std::endl;
            return false;
        }
    }
    return true;
}

bool my_householder(VectorXd a) {
    std::cout << a << std::endl;

    HouseholderData<double> hh_data = make_householder_vector_like_eigen<double>(a);

    std::cout << "hh: " << hh_data.hh_tail.transpose() << std::endl;
    std::cout << "beta: " << hh_data.beta << std::endl;
    std::cout << "tau: " << hh_data.tau << std::endl;

    std::cout << "tmp: " << hh_data.hh_tail.adjoint() * a.tail(3) << std::endl;
    std::cout << "tmp: " << a(0) + hh_data.hh_tail.adjoint() * a.tail(3) << std::endl;

    std::cout << "first: " << a(0) - hh_data.tau * (a(0) + hh_data.hh_tail.adjoint() * a.tail(3)) << std::endl;
    std::cout << "rest: " << a.tail(3) - hh_data.tau * hh_data.hh_tail * (a(0) + hh_data.hh_tail.adjoint() * a.tail(3)) << std::endl;

    apply_householder_like_eigen<double>(a, hh_data);

    std::cout << a << std::endl;

    if (std::abs(std::abs(a(0)) - std::sqrt(30)) > 0.1) {
        std::cout << "first element is not sqrt(30)" << std::endl;
        return false;
    }

    for (Eigen::Index idx = 1; idx < a.size(); idx++) {
        if (std::abs(a(idx)) > 0.1) {
            std::cout << idx << ". element is not sqrt(30)" << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // test householder ortho
    VectorXd a = VectorXd::Zero(4);
    a(0) = 1;
    a(1) = 2;
    a(2) = 3;
    a(3) = 4;
    if (!eigen_householder(a)) {
        return -1;
    }

    a = VectorXd::Zero(4);
    a(0) = 1;
    a(1) = 2;
    a(2) = 3;
    a(3) = 4;
    if (!my_householder(a)) {
        return -1;
    }
}
