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

    inline Eigen::Matrix<double, 4, 4> SE3ExpMap(const Eigen::Matrix<double, 6, 1>& se3) {
        Eigen::Vector<double, 3> omega_vector = se3.head(3);
        Eigen::Vector<double, 3> u_vector = se3.tail(3);
        double theta = sqrt(omega_vector.dot(omega_vector));
        Eigen::Matrix<double, 3, 3> omega_vec_skew = Skew(omega_vector);

        // Calculate coefficients in the formula
        double a1;
        double a2;
        double a3;

        if (theta < 1e-7) {
            a1 = 1;
            a2 = 0.5;
            a3 = 1.0 / 6.0;
        } else {
            a1 = sin(theta) / theta;
            a2 = (1 - cos(theta)) / (theta * theta);
            a3 = (1 - a1) / (theta * theta);
        }

        // Define identity matrix and V matrix to complete the formula
        Eigen::Matrix<double, 3, 3> identity_mat_3x3 = Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, 3, 3> v_mat = identity_mat_3x3 + a2 * omega_vec_skew +
                                       a3 * omega_vec_skew * omega_vec_skew;

        Eigen::Matrix<double, 4, 4> se3_exp_rot_mat = Eigen::Matrix4d::Zero();
        se3_exp_rot_mat.block(0, 0, 3, 3) = identity_mat_3x3 + a1 * omega_vec_skew +
                                            a2 * omega_vec_skew * omega_vec_skew;
        se3_exp_rot_mat.block(0, 3, 3, 1) = v_mat * u_vector;
        se3_exp_rot_mat(3, 3) = 1;

        return se3_exp_rot_mat;
    }

    inline Eigen::Matrix<double, 6, 1> se3LogMap(const Eigen::Matrix<double, 4, 4>& mat) {
//        // @todo Solve a problem of eigen block malfunction inside a function defined
//        // with template.
//        Eigen::Matrix<double, 3, 3> SE3_rot;
//        SE3_rot << SE3(0, 0), SE3(0, 1), SE3(0, 2), SE3(1, 0), SE3(1, 1), SE3(1, 2),
//                SE3(2, 0), SE3(2, 1), SE3(2, 2);
//
//        Eigen::Vector<double, 3> so3_log_vector = so3LogMap(SE3_rot);
//        const double so3_log_vector_norm = so3_log_vector.norm();
//
//        Eigen::Vector<double, 3> transition;
//        transition << SE3(0, 3), SE3(1, 3), SE3(2, 3);
//
//        Eigen::Matrix<double, 6, 1> se3_log;
//        if (so3_log_vector_norm < 1e-10) {
//            se3_log << so3_log_vector, transition;
//            return se3_log;
//        } else {
//            Eigen::Matrix<double, 3, 3> so3_log_skew_mat =
//                    Skew(so3_log_vector / so3_log_vector_norm);
//            double tangential = tan(0.5 * so3_log_vector_norm);
//            Eigen::Vector<double, 3> so3_log_skew_mat_by_transition =
//                    so3_log_skew_mat * transition;
//            Eigen::Vector<double, 3> u_vector =
//                    transition -
//                    (0.5 * so3_log_vector_norm) * so3_log_skew_mat_by_transition +
//                    (1 - so3_log_vector_norm / (2. * tangential)) *
//                    (so3_log_skew_mat * so3_log_skew_mat_by_transition);
//            se3_log << so3_log_vector, u_vector;
//            return se3_log;
//        }
        Eigen::Vector3d w = so3LogMap(mat.block<3, 3>(0, 0));
        Eigen::Vector3d T = mat.block<3, 1>(0, 3);
        const double t = w.norm();
        if (t < 1e-10) {
            Eigen::Matrix<double, 6, 1> log;
            log << w, T;
            return log;
        } else {
            Eigen::Matrix3d W = Skew(w / t);
            // Formula from Agrawal06iros, equation (14)
            // simplified with Mathematica, and multiplying in T to avoid matrix math
            double Tan = tan(0.5 * t);
            Eigen::Vector3d WT = W * T;
            Eigen::Vector3d u = T - (0.5 * t) * WT + (1 - t / (2. * Tan)) * (W * WT);
            Eigen::Matrix<double, 6, 1> log;
            log << w, u;
            return log;
        }
    }



}
#endif //MSCKF_STUDY_TRANSFORMATION_H
