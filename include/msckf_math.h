//
// Created by ilseo on 11/23/22.
//
#include <Eigen/Eigen>

#ifndef MSCKF_STUDY_TRANSFORMATION_H_
#define MSCKF_STUDY_TRANSFORMATION_H_

namespace msckf_math {
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

}
#endif //MSCKF_STUDY_TRANSFORMATION_H
