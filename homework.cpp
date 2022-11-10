#include <Eigen/Eigen>
#include <math.h>

#include <iostream>
#include <iterator>
#include <random>

inline Eigen::Matrix<double, 3, 1> SO3LogMap(const Eigen::Matrix<double, 3, 3>& SO3) {
Eigen::Matrix<double, 3, 1> so3;
double theta = acos((SO3.trace() - 1)/2);
Eigen::Matrix<double, 3, 1> tmp_vec(SO3(2, 1) - SO3(1, 2), SO3(0, 2) - SO3(2, 0), SO3(1, 0) - SO3(0, 1));
so3 = 1/(2*sin(theta)) * tmp_vec;

return so3;
}

inline Eigen::Matrix<double, 3, 3> so3ExpMap(const Eigen::Matrix<double, 3, 1>& so3) {
Eigen::Matrix<double, 3, 3> SO3;

Eigen::Matrix<double, 3, 3> skew;
skew << 0, -so3(2), so3(1),
so3(2), 0, -so3(0),
-so3(1), so3(0), 0;

double theta = acos((SO3.trace() - 1)/2);

SO3 = Eigen::Matrix3d::Identity() + sin(theta) * skew + (1 - cos(theta)) * skew * skew;

return SO3;
}

int main(int argc, char** argv) {
Eigen::Matrix<double, 3, 3> SO3;
SO3 << -1, 0, 0, 
0, 1, 0, 
0, 0, -1;

Eigen::Matrix<double, 3, 1> so3 = SO3LogMap(SO3);
std::cout << "so3 : " << std::endl;
std::cout << so3 << std::endl;

Eigen::Matrix<double, 3, 3> SO3_recomputed = so3ExpMap(so3);
std::cout << "SO3 : " << std::endl;
std::cout << SO3_recomputed << std::endl;

Eigen::Matrix<double, 6, 1> se3;
se3 << so3, 10, 0, 0;
std::cout << "se3 : " << std::endl;
std::cout << se3 << std::endl;

Eigen::Matrix<double, 6, 1> initial_pose;
// Define random generator with Gaussian distribution
const double mean = 0.0;
const double stddev = 0.1;
const unsigned int scale = 0.5;
std::default_random_engine generator;
std::normal_distribution<double> dist(mean, stddev);

// Add Gaussian noise
for (auto& x : initial_pose) {
    x = x + (scale * dist(generator));
}
std::cout << "Initial Guess : " << std::endl;
std::cout << initial_pose << std::endl;
}