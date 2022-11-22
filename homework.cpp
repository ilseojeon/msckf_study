#include <Eigen/Eigen>
#include <math.h>

#include <iostream>
#include <iterator>
#include <random>

inline Eigen::Matrix3d Skew(const Eigen::Vector3d& vec) {
    Eigen::Matrix3d skew_mat;

    skew_mat << 0, -vec(2), vec(1),
vec(2), 0, -vec(0),
-vec(1), vec(0), 0;

    return skew_mat;
}

inline Eigen::Matrix<double, 3, 1> so3LogMap(const Eigen::Matrix<double, 3, 3>& SO3) {
    Eigen::Matrix<double, 3, 1> so3;
    double trace = SO3.trace();

    // when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
    // we do something special
    if (trace + 1.0 < 1e-10) {
        if (std::abs(SO3(2, 2) + 1.0) > 1e-5)
            so3 = (M_PI / sqrt(2.0 + 2.0 * SO3(2, 2))) * Eigen::Vector3d(SO3(0, 2), SO3(1, 2), 1.0 + SO3(2, 2));
        else if (std::abs(SO3(1, 1) + 1.0) > 1e-5)
            so3 = (M_PI / sqrt(2.0 + 2.0 * SO3(1, 1))) * Eigen::Vector3d(SO3(0, 1), 1.0 + SO3(1, 1), SO3(2, 1));
        else
            // if(std::abs(R.r1_.x()+1.0) > 1e-5)  This is implicit
            so3 = (M_PI / sqrt(2.0 + 2.0 * SO3(0, 0))) * Eigen::Vector3d(1.0 + SO3(0, 0), SO3(1, 0), SO3(2, 0));
    } else {
        double magnitude;
        const double tr_3 = trace - 3.0; // always negative
        if (tr_3 < -1e-7) {
            double theta = acos((trace - 1.0) / 2.0);
            magnitude = theta / (2.0 * sin(theta));
        } else {
            // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            // use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            // see https://github.com/borglab/gtsam/issues/746 for details
            magnitude = 0.5 - tr_3 / 12.0;
        }
        so3 = magnitude * Eigen::Vector3d(SO3(2, 0) - SO3(1, 2), SO3(0, 2) - SO3(2, 0), SO3(1, 0) - SO3(0, 1));
    }
    return so3;
}

inline Eigen::Matrix<double, 3, 3> SO3ExpMap(const Eigen::Matrix<double, 3, 1>& so3) {
    Eigen::Matrix<double, 3, 3> SO3;

    Eigen::Matrix<double, 3, 3> skew = Skew(so3);

    double theta = so3.norm();
    double a1;
    double a2;

    if (theta < 1e-7) {
        a1 = 1;
        a2 = 0.5;
    } else {
        a1 = sin(theta) / theta;
        a2 = (1 - cos(theta)) / (theta * theta);
    }

    if (theta == 0) {
        SO3 = Eigen::Matrix3d::Identity();
    } else {
        SO3 = Eigen::Matrix3d::Identity() + a1 * skew + a2 * skew * skew;
    }

    return SO3;
}

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
    R = SO3ExpMap(state.segment(0, 3));
    t = state.segment(3, 3);

    // e = (C1_R_C2 * C2_pt + C1_t_C2) - C1_pt
    res = (R * pt_prime + t) - pt;

    return res;
}

inline Eigen::Matrix<double, 3, 6> CalJacobian(const Eigen::Matrix<double, 6, 1>& state, const Eigen::Matrix<double, 3, 1>& pt_prime) {
    Eigen::Matrix<double, 3, 6> jacobian;
    
    // J = [-[C1_R_C2 * C2_p]x  I] (3 x 6 mat)
    jacobian.block(0, 0, 3, 3) = -Skew(SO3ExpMap(state.segment(0, 3)) * pt_prime);
    jacobian.block(0, 3, 3, 3) = Eigen::Matrix3d::Identity(); 

    return jacobian;
}

inline Eigen::Matrix<double, 6, 1> Caldx(const Eigen::MatrixXd& jacobian, const Eigen::MatrixXd& residual) {
    Eigen::Matrix<double, 6, 1> dx;
    // dx = (J^T * J)^(-1) * J^T * res
    Eigen::MatrixXd jacobiant_by_jacobian_inverse;
    jacobiant_by_jacobian_inverse = (jacobian.transpose() * jacobian).inverse();
    dx = -jacobiant_by_jacobian_inverse * jacobian.transpose() * residual;

    return dx;
}

int main(int argc, char** argv) {
// C1_R_C2
Eigen::Matrix<double, 3, 3> SO3;
SO3 << -1, 0, 0, 
0, 1, 0, 
0, 0, -1;

// HW 1 Get se3 vector
Eigen::Matrix<double, 3, 1> so3 = so3LogMap(SO3);
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
    x = x + AddNoise(0.0, 0.05, 0.005);
}
for (auto& x : initial_pose.segment(3, 3)) {
    x = x + AddNoise(0.0, 0.1, 0.001);
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
Eigen::MatrixXd res;
res.resize(3 * num_of_pts, 1);
for (unsigned int i = 0; i < num_of_pts; i++) {
    auto tmp = CalRes(initial_pose, pts.at(i), pts_prime.at(i));
    res.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
}

// HW 6 Calculate jacobian and stack them
Eigen::MatrixXd jacobian;
jacobian.resize(3 * num_of_pts, 6);
std::cout << "jacobian" << std::endl;
for (unsigned int i = 0; i < num_of_pts; i++) {
    auto tmp = CalJacobian(initial_pose, pts_prime.at(i));
    jacobian.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
}
std::cout << jacobian << std::endl;

// HW 7 Calculate dx
auto dx = Caldx(jacobian, res);
std::cout << "dx" << std::endl;
std::cout << dx.transpose() << std::endl;

// HW 8 Calculate new residual
//Eigen::Matrix<double, 6, 1> new_pose;
//new_pose.segment(0, 3) = so3LogMap(SO3ExpMap(initial_pose.segment(0, 3)) * SO3ExpMap(dx.segment(0, 3)));
//new_pose.segment(3, 3) = initial_pose.segment(3, 3) + dx.segment(3, 3);
//std::cout << "new pose : " << std::endl;
//std::cout << new_pose.transpose() << std::endl;
//
//Eigen::MatrixXd new_res;
//new_res.resize(3 * num_of_pts, 1);
//for (unsigned int i = 0; i < num_of_pts; i++) {
//    auto tmp = CalRes(new_pose, pts.at(i), pts_prime.at(i));
//    new_res.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
//}
//std::cout << "residual old - residual new" << std::endl;
//std::cout << res.transpose() * res<< std::endl;
//std::cout << new_res.transpose() * new_res << std::endl;

// HW 9 Iteration
Eigen::Matrix<double, 6, 1> new_pose;
Eigen::MatrixXd old_res;
old_res.resize(3 * num_of_pts, 1);
old_res = res;
Eigen::MatrixXd new_res;
new_res.resize(3 * num_of_pts, 1);
Eigen::MatrixXd res_diff;
unsigned int kNumOfIteration = 100;
unsigned int iteration = 0;
const double kDelta = 1e-6;
while (iteration < kNumOfIteration) {
    iteration++;
    new_pose.segment(0, 3) = so3LogMap(SO3ExpMap(initial_pose.segment(0, 3)) * SO3ExpMap(dx.segment(0, 3)));
    new_pose.segment(3, 3) = initial_pose.segment(3, 3) + dx.segment(3, 3);

    for (unsigned int i = 0; i < num_of_pts; i++) {
        auto tmp = CalRes(new_pose, pts.at(i), pts_prime.at(i));
        new_res.block(3 * i, 0, tmp.rows(), tmp.cols()) = tmp;
    }
    res_diff = old_res - new_res;
    old_res = new_res;

    if (res_diff.norm() < kDelta)  {
        std::cout << "Completed Iteration" << std::endl;
        std::cout << "Num of iteration : " << iteration << std::endl;
        std::cout << "New pose : " << new_pose.transpose() << std::endl;
        break;
    }

    if (res.norm() < new_res.norm()) {
        std::cout << "Diverged" << std::endl;
        break;
    }
}

return 0;
}