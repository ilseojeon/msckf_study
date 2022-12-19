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

inline Eigen::Matrix<double, 3, 1> CalP2PlRes(const Eigen::Matrix<double, 6, 1>& state, const Eigen::Matrix<double, 3, 1>& pt, const Eigen::Matrix<double, 3, 1>& pt_prime) {

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
        auto tmp = C1_R_C2.transpose() * (pt - se3.segment(3, 3));
        pts_prime.emplace_back(tmp);
    }

    Eigen::MatrixXd initial_p2p_res;
    initial_p2p_res.resize(3 * num_of_pts, 1);
    for (unsigned int i = 0; i < num_of_pts; i++) {
        auto tmp = CalP2PRes(initial_pose, pts.at(i), pts_prime.at(i));
        initial_p2p_res.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
    }

    // HW 11 GT points on the plane
    std::vector<Eigen::Matrix<double, 3, 1>> pts_plane;
    pts_plane.reserve(num_of_pts);
    pts_plane.emplace_back(1, 1, 2);
    Eigen::Vector3d noise_vector(AddNoise(0.0, 0.05, 0.01), AddNoise(0.0, 0.05, 0.01), AddNoise(0.0, 0.05, 0.01));
    Eigen::Vector3d noise_vector2(AddNoise(0.0, 0.05, 0.01), AddNoise(0.0, 0.05, 0.01), AddNoise(0.0, 0.05, 0.01));
    pts_plane.at(1) = pts_plane.at(0) + noise_vector;
    pts_plane.at(2) = pts_plane.at(1) + noise_vector2;

    Eigen::VectorXd normal_vecotr;
    normal_vecotr = (pts_plane.at(0) - pts_plane.at(1)).cross(pts_plane.at(0) - pts_plane.at(2));
}