#include <Eigen/Eigen>

#include <iostream>
#include <random>

#include "msckf_math.h"
#include "optimizer.h"

inline double AddNoise(double mean, double stddev, double scale) {
    // Define random generator with Gaussian distribution
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, stddev);
    return scale * dist(generator);
}

inline Eigen::Matrix<double, 3, 1> CalRes(const Eigen::Matrix<double, 6, 1>& state, const Eigen::Matrix<double, 3, 1>& pt, const Eigen::Matrix<double, 3, 1>& pt_prime) {
    Eigen::Matrix<double, 3, 1> res;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    R = msckf_math::SO3ExpMap(state.segment(0, 3));
    t = state.segment(3, 3);

    // e = (C1_R_C2 * C2_pt + C1_t_C2) - C1_pt
    res = (R * pt_prime + t) - pt;

    return res;
}

inline Eigen::Matrix<double, 3, 6> CalLeftJacobian(const Eigen::Matrix<double, 6, 1>& state, const Eigen::Matrix<double, 3, 1>& pt_prime) {
    Eigen::Matrix<double, 3, 6> jacobian;

    // J = [-[C1_R_C2 * C2_p]x  I] (3 x 6 mat)
    jacobian.block(0, 0, 3, 3) = -msckf_math::Skew(msckf_math::SO3ExpMap(state.segment(0, 3)) * pt_prime);
    jacobian.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();

    return jacobian;
}

inline Eigen::Matrix<double, 3, 6> CalRightJacobian(const Eigen::Matrix<double, 6, 1>& state, const Eigen::Matrix<double, 3, 1>& pt_prime) {
    Eigen::Matrix<double, 3, 6> jacobian;

    // J = [-[C1_R_C2 * C2_p]x  I] (3 x 6 mat)
    jacobian.block(0, 0, 3, 3) = -msckf_math::Skew(msckf_math::SO3ExpMap(-state.segment(0, 3)) * pt_prime);
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

inline Eigen::Matrix<double, 6, 1> Update(Eigen::Matrix<double, 6, 1>& pose, const Eigen::Matrix<double, 6, 1>& dx) {
    Eigen::Matrix<double, 6, 1> new_pose;
    new_pose.segment(0, 3) = msckf_math::so3LogMap(msckf_math::SO3ExpMap(dx.segment(0, 3)) * msckf_math::SO3ExpMap(pose.segment(0, 3)));
    new_pose.segment(3, 3) = pose.segment(3, 3) + dx.segment(3, 3);
    return new_pose;
}

int main(int argc, char** argv) {
    // C1_R_C2
    Eigen::Matrix<double, 3, 3> SO3;
    SO3 << -1, 0, 0,
    0, 1, 0,
    0, 0, -1;

    // HW 1 Get se3 vector
    Eigen::Matrix<double, 3, 1> so3 = msckf_math::so3LogMap(SO3);
    std::cout << "so3 : " << std::endl;
    std::cout << so3.transpose() << std::endl;

    // C1_se3_C2
    Eigen::Matrix<double, 6, 1> se3;
    se3 << so3, 10, 0, 0;
    std::cout << "se3 : " << std::endl;
    std::cout << se3.transpose() << std::endl;

    // HW 2 Get initial pose (state)
    Eigen::Matrix<double, 6, 1> initial_pose(se3);
    for (auto& x : initial_pose.segment(0, 3)) {
        x = x + AddNoise(0.0, 0.005, 0.1);
    }
    for (auto& x : initial_pose.segment(3, 3)) {
        x = x + AddNoise(0.0, 0.05, 0.5);
    }

    std::cout << "Initial Guess : " << std::endl;
    std::cout << initial_pose.transpose() << std::endl;

    // HW 3
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
        auto tmp = SO3.transpose() * (pt - se3.segment(3, 3));
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

    // HW 5 Calculate residual and stack them
    Eigen::MatrixXd initial_res;
    initial_res.resize(3 * num_of_pts, 1);
    for (unsigned int i = 0; i < num_of_pts; i++) {
        auto tmp = CalRes(initial_pose, pts.at(i), pts_prime.at(i));
        initial_res.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
    }
    std::cout << "initial residual : " << std::endl;
    std::cout << initial_res.transpose() << std::endl;

    // HW 6 Calculate jacobian and stack them
    Eigen::MatrixXd initial_jacobian;
    initial_jacobian.resize(3 * num_of_pts, 6);
    std::cout << "jacobian" << std::endl;
    for (unsigned int i = 0; i < num_of_pts; i++) {
        auto tmp = CalLeftJacobian(initial_pose, pts_prime.at(i));
        initial_jacobian.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
    }
    std::cout << initial_jacobian << std::endl;

    // HW 7 Calculate dx
    auto initial_dx = Caldx(initial_jacobian, initial_res);
    std::cout << "dx" << std::endl;
    std::cout << initial_dx.transpose() << std::endl;

    // HW 8 Update
    Eigen::Matrix<double, 6, 1> new_pose;
    new_pose = Update(initial_pose, initial_dx);
    std::cout << "new pose : " << std::endl;
    std::cout << new_pose.transpose() << std::endl;

//    Eigen::MatrixXd new_res;
//    new_res.resize(3 * num_of_pts, 1);
//    for (unsigned int i = 0; i < num_of_pts; i++) {
//        auto tmp = CalRes(new_pose, pts.at(i), pts_prime.at(i));
//        new_res.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
//    }
//    std::cout << "residual old - residual new" << std::endl;
//    std::cout << initial_res.transpose() * initial_res<< std::endl;
//    std::cout << new_res.transpose() * new_res << std::endl;

    // HW 9 Iteration
    // Declare initial values
    Eigen::MatrixXd old_res(initial_res);
    Eigen::Matrix<double, 6, 1> pose(new_pose);

    Eigen::MatrixXd new_res;
    new_res.resize(3 * num_of_pts, 1);

    Eigen::MatrixXd res_diff;
    Eigen::MatrixXd new_jacobian;
    new_jacobian.resize(3 * num_of_pts, 6);
    Eigen::MatrixXd new_dx;

    const unsigned int kNumOfIteration = 100;
    unsigned int iteration = 0;
    const double kDelta = 1e-6;

    while (iteration < kNumOfIteration) {
        for (unsigned int i = 0; i < num_of_pts; i++) {
            auto tmp = CalRes(pose, pts.at(i), pts_prime.at(i));
            new_res.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
        }

        for (unsigned int i = 0; i < num_of_pts; i++) {
            auto tmp = CalLeftJacobian(pose, pts_prime.at(i));
            new_jacobian.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
        }
        new_dx = Caldx(new_jacobian, new_res);
        new_pose = Update(pose, new_dx);

        res_diff = old_res - new_res;
        old_res = new_res;

        if (res_diff.norm() < kDelta)  {
            std::cout << "Completed Iteration" << std::endl;
            std::cout << "Num of iteration : " << iteration << std::endl;
            std::cout << "New pose : " << new_pose.transpose() << std::endl;
            break;
        }

        if (old_res.norm() < new_res.norm()) {
            std::cout << "Diverged" << std::endl;
            break;
        }
        iteration++;
    }

    return 0;
}