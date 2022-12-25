#include <Eigen/Eigen>

#include <iostream>
#include <random>

#include "msckf_math.h"

inline double AddNoise(double mean, double stddev, double scale) {
    // Define random generator with Gaussian distribution
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, stddev);
    return scale * dist(generator);
}

inline Eigen::Matrix<double, 3, 1> CalP2PRes(const Eigen::Matrix<double, 6, 1>& state, const Eigen::Matrix<double, 3, 1>& pt, const Eigen::Matrix<double, 3, 1>& pt_prime) {
    Eigen::Matrix<double, 3, 1> res;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    R = msckf_math::SO3ExpMap(state.segment(0, 3));
    t = state.segment(3, 3);

    // e = (C1_R_C2 * C2_pt + C1_t_C2) - C1_pt
    res = (R * pt_prime + t) - pt;

    return res;
}

inline Eigen::Matrix<double, 1, 1> CalP2PlRes(const Eigen::Matrix<double, 6, 1>& state, const Eigen::Matrix<double, 3, 1>& pt, const Eigen::Matrix<double, 3, 1>& pt2, const Eigen::Matrix<double, 3, 1>& normal_vector) {
    Eigen::Matrix<double, 1, 1> res;
    res = normal_vector.transpose() * (msckf_math::SO3ExpMap(state.segment(0, 3)) * pt + state.segment(3, 3) - pt2);

    return res;
}

inline Eigen::Matrix<double, 3, 6> CalP2PJacobian(const Eigen::Matrix<double, 6, 1>& state, const Eigen::Matrix<double, 3, 1>& pt_prime) {
    Eigen::Matrix<double, 3, 6> jacobian;

    // J = [-[C1_R_C2 * C2_p]x  I] (3 x 6 mat)
    jacobian.block(0, 0, 3, 3) = -msckf_math::Skew(msckf_math::SO3ExpMap(state.segment(0, 3)) * pt_prime);
    jacobian.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();

    return jacobian;
}

inline Eigen::Matrix<double, 1, 6> CalP2PlJacobian(const Eigen::Matrix<double, 6, 1>& state, const Eigen::Matrix<double, 3, 1>& pt, const Eigen::Matrix<double, 3, 1> normal_vec) {
    Eigen::Matrix<double, 1, 6> jacobian;
    Eigen::Matrix<double, 3, 6> jacobian_mid;
    jacobian_mid.block(0, 0, 3, 3) = -msckf_math::Skew(msckf_math::SO3ExpMap(state.segment(0, 3)) * pt);
    jacobian_mid.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity();
    jacobian = normal_vec.transpose() * jacobian_mid;

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
    Eigen::Matrix<double, 3, 3> C1_R_C2;
    C1_R_C2 << -1, 0, 0,
            0, 1, 0,
            0, 0, -1;

    Eigen::Matrix<double, 3, 1> so3 = msckf_math::so3LogMap(C1_R_C2);
    Eigen::Matrix<double, 6, 1> se3;
    se3 << so3, 10, 0, 0;

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

    unsigned int num_of_pts = 3;
    unsigned int num_of_normal_vec = 1;
    std::vector<Eigen::Matrix<double, 3, 1>> pts;
    std::vector<Eigen::Matrix<double, 3, 1>> pts_prime;
    pts.reserve(num_of_pts);
    pts.emplace_back(0, 1, 2);
    pts.emplace_back(0, 2, 3);
    pts.emplace_back(3, 4, 2);
    pts_prime.reserve(num_of_pts);

    // pt' = C2_R_C1 * (C1_pt - C1_t_C2)
    for (auto& pt : pts) {
        auto tmp = C1_R_C2.transpose() * (pt - se3.segment(3, 3));
        pts_prime.emplace_back(tmp);
    }

    Eigen::MatrixXd initial_p2p_res;
    initial_p2p_res.resize(3 * num_of_pts + num_of_normal_vec, 1);
    for (unsigned int i = 0; i < num_of_pts; i++) {
        auto tmp = CalP2PRes(initial_pose, pts.at(i), pts_prime.at(i));
        initial_p2p_res.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
    }

    Eigen::MatrixXd initial_jacobian;
    initial_jacobian.resize(3 * num_of_pts + num_of_normal_vec, 6);
    for (unsigned int i = 0; i < num_of_pts; i++) {
        auto tmp = CalP2PJacobian(initial_pose, pts_prime.at(i));
        initial_jacobian.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
    }

    // HW 11 GT points on the plane
    std::vector<Eigen::Matrix<double, 3, 1>> pts_plane;
    pts_plane.resize(num_of_pts);
    pts_plane.at(0) = Eigen::Vector3d(1, 1, 2);
    Eigen::Vector3d noise_vector(AddNoise(0.0, 0.05, 0.01), AddNoise(0.0, 0.5, 0.1), AddNoise(0.0, 0.05, 0.01));
    Eigen::Vector3d noise_vector2(AddNoise(0.0, 0.5, 0.1), AddNoise(0.0, 0.05, 0.01), AddNoise(0.0, 0.5, 0.1));
    pts_plane.at(1) = pts_plane.at(0) + noise_vector;
    pts_plane.at(2) = pts_plane.at(0) + noise_vector2;

    Eigen::Vector3d normal_vector;
    normal_vector = (pts_plane.at(0) - pts_plane.at(1)).cross(pts_plane.at(0) - pts_plane.at(2));
    normal_vector = normal_vector / normal_vector.norm();

    Eigen::Matrix<double, 3, 1> pts_plane_prime;
    pts_plane_prime = C1_R_C2.transpose() * (pts_plane.at(0) - se3.segment(3, 3));

    // Stack joint residual, joint jacobian
    initial_p2p_res.block(9, 0, 1, 1) = CalP2PlRes(initial_pose, pts_plane_prime, pts_plane.at(0), normal_vector);
    initial_jacobian.block(9, 0, 1, 6) = CalP2PlJacobian(initial_pose, pts_plane_prime, normal_vector);

    std::cout << "initial residual : " << std::endl;
    std::cout << initial_p2p_res.transpose() << std::endl;
    std::cout << "jacobian" << std::endl;
    std::cout << initial_jacobian << std::endl;

    auto initial_dx = Caldx(initial_jacobian, initial_p2p_res);
    std::cout << "dx" << std::endl;
    std::cout << initial_dx.transpose() << std::endl;

    Eigen::Matrix<double, 6, 1> new_pose;
    new_pose = Update(initial_pose, initial_dx);
    std::cout << "new pose : " << std::endl;
    std::cout << new_pose.transpose() << std::endl;

    // Iteration
    Eigen::MatrixXd old_res(initial_p2p_res);
    Eigen::Matrix<double, 6, 1> pose(new_pose);

    Eigen::MatrixXd new_res;
    new_res.resize(3 * num_of_pts + num_of_normal_vec, 1);

    Eigen::MatrixXd res_diff;
    Eigen::MatrixXd new_jacobian;
    new_jacobian.resize(3 * num_of_pts + num_of_normal_vec, 6);
    Eigen::MatrixXd new_dx;

    const unsigned int kNumOfIteration = 100;
    unsigned int iteration = 0;
    const double kDelta = 1e-6;

    while (iteration < kNumOfIteration) {
        for (unsigned int i = 0; i < num_of_pts; i++) {
            auto tmp = CalP2PRes(pose, pts.at(i), pts_prime.at(i));
            new_res.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
        }
        new_res.block(9, 0, 1, 1) = CalP2PlRes(pose, pts_plane_prime, pts_plane.at(0), normal_vector);

        for (unsigned int i = 0; i < num_of_pts; i++) {
            auto tmp = CalP2PJacobian(pose, pts_prime.at(i));
            new_jacobian.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
        }
        new_jacobian.block(9, 0, 1, 6) = CalP2PlJacobian(initial_pose, pts_plane_prime, normal_vector);

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