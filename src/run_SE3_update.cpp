#include <Eigen/Eigen>

#include <iostream>
#include <random>

#include "msckf_math.h"

namespace SE3_updater {
    inline double AddNoise(double mean, double stddev, double scale) {
        // Define random generator with Gaussian distribution
        std::default_random_engine generator;
        std::normal_distribution<double> dist(mean, stddev);
        return scale * dist(generator);
    }

    inline Eigen::Matrix<double, 3, 1> CalRes(const Eigen::Matrix<double, 6, 1>& state, const Eigen::Matrix<double, 3, 1>& C1_pt, const Eigen::Matrix<double, 3, 1>& C2_pt) {
        Eigen::Matrix<double, 3, 1> res;
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        R = msckf_math::SO3ExpMap(state.segment(0, 3));
        t = state.segment(3, 3);

        // e = (C1_R_C2 * C2_pt + C1_t_C2) - C1_pt
        res = (R * C2_pt + t) - C1_pt;

        return res;
    }

    inline Eigen::Matrix<double, 3, 6> CalLeftJacobian(const Eigen::Matrix<double, 6, 1>& state, const Eigen::Matrix<double, 3, 1>& C2_pt) {
        Eigen::Matrix<double, 3, 6> jacobian;

        // J = [-Skew(C1_R_C2 * C2_pt + C1_t_C2)     I] (3 x 6 mat)
        jacobian.block(0, 0, 3, 3) = -msckf_math::Skew(msckf_math::SO3ExpMap(state.segment(0, 3)) * C2_pt + state.segment(3, 3));
        jacobian.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();

        return jacobian;
    }

    inline Eigen::Matrix<double, 6, 1> Caldx(const Eigen::MatrixXd& jacobian, const Eigen::MatrixXd& residual) {
        Eigen::Matrix<double, 6, 1> dx;
        // dx = -(J^T * J)^(-1) * J^T * res
        Eigen::MatrixXd jacobiant_by_jacobian_inverse;
        jacobiant_by_jacobian_inverse = (jacobian.transpose() * jacobian).inverse();
        dx = -jacobiant_by_jacobian_inverse * jacobian.transpose() * residual;

        return dx;
    }

    inline Eigen::Matrix<double, 6, 1> Update(Eigen::Matrix<double, 6, 1>& state, const Eigen::Matrix<double, 6, 1>& dx) {
        Eigen::Matrix<double, 6, 1> new_state;
        // x_new (rot) = log(exp(dx) * exp(x))
        // x_new (trans) = exp(dx) * exp(x) + dx
        new_state.segment(0, 3) = msckf_math::so3LogMap(msckf_math::SO3ExpMap(dx.segment(0, 3)) * msckf_math::SO3ExpMap(state.segment(0, 3)));
        new_state.segment(3, 3) = msckf_math::SO3ExpMap(dx.segment(0, 3)) * state.segment(3, 3) + dx.segment(3, 3);

//        new_state = msckf_math::se3LogMap(msckf_math::SE3ExpMap(dx) * msckf_math::SE3ExpMap(state));

        return new_state;
    }
};

int main(int argc, char** argv) {
    Eigen::Matrix<double, 4, 4> C1_T_C2;
    C1_T_C2 << -1, 0, 0, 10,
                0, 1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, 1;
    Eigen::Matrix<double, 6, 1> C1_xi_C2;
    C1_xi_C2.segment(0, 3) = msckf_math::so3LogMap(C1_T_C2.block(0, 0, 3, 3));
    C1_xi_C2.segment(3, 3) = C1_T_C2.block(0, 3, 3, 1);
    std::cout << "SE3 : " << std::endl;
    std::cout << C1_T_C2 << std::endl;
    std::cout << "se3 : " << std::endl;
    std::cout << C1_xi_C2.transpose() << std::endl;

    // Get initial pose with noise (state)
    Eigen::Matrix<double, 6, 1> initial_pose(C1_xi_C2);
    for (auto& x : initial_pose.segment(0, 3)) {
        x = x + SE3_updater::AddNoise(0.0, 0.001, 0.1);
    }
    for (auto& x : initial_pose.segment(3, 3)) {
        x = x + SE3_updater::AddNoise(0.0, 0.01, 0.1);
    }

    std::cout << "Initial Guess : " << std::endl;
    std::cout << initial_pose.transpose() << std::endl;

    unsigned int num_of_pts = 3;
    std::vector<Eigen::Matrix<double, 3, 1>> pts;
    std::vector<Eigen::Matrix<double, 3, 1>> pts_prime;
    pts.reserve(num_of_pts);
    pts.emplace_back(0, 1, 2);
    pts.emplace_back(0, 2, 3);
    pts.emplace_back(3, 4, 2);
    pts_prime.reserve(num_of_pts);

    // pt' = C2_R_C1 * (C1_pt - C1_t_C2)
    for (auto& pt : pts) {
        auto tmp = msckf_math::SO3ExpMap(initial_pose.segment(0, 3)).transpose() * (pt - initial_pose.segment(3, 3));
        pts_prime.emplace_back(tmp);
    }
    std::cout << "C1_pts : " << std::endl;
    std::cout << pts.at(0).transpose() << std::endl;
    std::cout << pts.at(1).transpose() << std::endl;
    std::cout << pts.at(2).transpose() << std::endl;
    std::cout << "C2_pts : " << std::endl;
    std::cout << pts_prime.at(0).transpose() << std::endl;
    std::cout << pts_prime.at(1).transpose() << std::endl;
    std::cout << pts_prime.at(2).transpose() << std::endl;

    // Calculate initial residual
    Eigen::MatrixXd initial_res;
    initial_res.resize(3 * num_of_pts, 1);
    for (unsigned int i = 0; i < num_of_pts; i++) {
        auto tmp = SE3_updater::CalRes(initial_pose, pts.at(i), pts_prime.at(i));
        initial_res.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
    }
    std::cout << "initial residual : " << std::endl;
    std::cout << initial_res.transpose() << std::endl;

    // Calculate initial jacobian
    Eigen::MatrixXd initial_jacobian;
    initial_jacobian.resize(3 * num_of_pts, 6);
    std::cout << "jacobian" << std::endl;
    for (unsigned int i = 0; i < num_of_pts; i++) {
        auto tmp = SE3_updater::CalLeftJacobian(initial_pose, pts_prime.at(i));
        initial_jacobian.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
    }
    std::cout << initial_jacobian << std::endl;

    // Calculate initial dx
    auto initial_dx = SE3_updater::Caldx(initial_jacobian, initial_res);
    std::cout << "dx" << std::endl;
    std::cout << initial_dx.transpose() << std::endl;

    // Update the state
    Eigen::Matrix<double, 6, 1> new_state;
    new_state = SE3_updater::Update(initial_pose, initial_dx);
    std::cout << "new state : " << std::endl;
    std::cout << new_state.transpose() << std::endl;

    // NEW RES
    Eigen::VectorXd new_res;
    new_res.resize(3 * num_of_pts, 1);
    for (unsigned int i = 0; i < num_of_pts; i++) {
        auto tmp = SE3_updater::CalRes(new_state, pts.at(i), pts_prime.at(i));
        new_res.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
    }
    std::cout << "residual old - residual new" << std::endl;
    std::cout << initial_res.transpose() * initial_res<< std::endl;
    std::cout << new_res.transpose() * new_res << std::endl;
}