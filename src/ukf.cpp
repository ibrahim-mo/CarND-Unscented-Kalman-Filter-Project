#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

//using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false;
  time_us_ = 0;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  weights_ = VectorXd(2 * n_aug_ + 1);

  //set weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i=1; i < 2 * n_aug_ + 1; i++)
    weights_(i) = 0.5 / (lambda_ + n_aug_);

  // set process noise parameters to realistic values
  // these values have been deriven from reading the ground truth data,
  // then observing the changes in v_acc = dv/dt and yaw_acc = dyaw_r/dt
  std_a_ = 0.5;
  std_yawdd_ = 0.5;

  // initialize NIS values
  NIS_laser_ = 0;
  NIS_radar_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */

    // first measurement
    std::cout << "UKF: " << std::endl;
    x_ << 1, 1, 1, 1, 1;

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      //set the state with the initial location; initilaize velocity, yaw, and yaw rate to 0's
      float px = meas_package.raw_measurements_[0];
      float py = meas_package.raw_measurements_[1];
      x_ << px, py, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      // Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      //set the state with the initial location; initilaize velocity, yaw, and yaw rate to 0's
      x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
    }

    //initialize process covariance matrix to the identity matrix
    P_ << 1,0,0,0,0,
        0,1,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,0,0,0,1;

    time_us_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // std::cout << "x_ (input) = " << std::endl << x_ << std::endl;
  // std::cout << "P_ (input) = " << std::endl << P_ << std::endl;

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  /**
    * Use elepased time - measured in seconds - to
    * predict the state mean and covariance matrix.
   **/
  
  //compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  // std::cout << "dt = " << dt << std::endl;

  Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    // Laser updates
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    // Radar updates
    UpdateRadar(meas_package);
  }

  // print the output
  std::cout << "x_ = " << std::endl << x_ << std::endl;
  std::cout << "P_ = " << std::endl << P_ << std::endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // std::cout << "---Prediction()---" << std::endl;

  /*
  ** 1. Generate augmented sigma points
  */

  //augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //augmented sigma points matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_; //long. accel. variance
  P_aug(n_x_+1, n_x_+1) = std_yawdd_ * std_yawdd_; //yaw accel. variance

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  //set first column of sigma point matrix
  Xsig_aug.col(0)  = x_aug;
  //set remaining sigma points
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i+1)     = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  //print result
  // std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  /*
  ** 2. Predict sigma points
  */
  
  for (int i=0; i < 2*n_aug_+1; i++) {
    VectorXd x_i = Xsig_aug.col(i);
    //extract values for better readability
    double v=x_i(2), psi=x_i(3), psid=x_i(4), nu_a=x_i(5), nu_psidd=x_i(6);

    //vectors to be added to predicted state
    VectorXd dx_i1(n_x_), dx_i2(n_x_);
    //avoid division by zero
    dx_i1 << ((psid != 0)? (v / psid) * (sin(psi + psid * delta_t) - sin(psi)) : v * cos(psi) * delta_t),
          ((psid != 0)? (v / psid) * (-cos(psi + psid * delta_t) + cos(psi)) : v * sin(psi) * delta_t),
          0,
          psid * delta_t,
          0;
    //add noise
    dx_i2 << 0.5 * delta_t * delta_t * cos(psi) * nu_a,
          0.5 * delta_t * delta_t * sin(psi) * nu_a,
          delta_t * nu_a,
          0.5 * delta_t * delta_t * nu_psidd,
          delta_t * nu_psidd;

    //write predicted sigma points into right column
    Xsig_pred_.col(i) = x_i.head(n_x_) + dx_i1 + dx_i2;
  }

  //print result
  // std::cout << "Xsig_pred_ = " << std::endl << Xsig_pred_ << std::endl;

  /*
  ** 3. Predict state mean and covariance matrix
  */

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose() ;
  }

  //print result
  // std::cout << "Predicted state" << std::endl;
  // std::cout << x_ << std::endl;
  // std::cout << "Predicted covariance matrix" << std::endl;
  // std::cout << P_ << std::endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // std::cout << "---UpdateLidar()---" << std::endl;

  /*
  ** 1. Predict lidar measurement
  */

  //set measurement dimension, laser can measure px, py
  int n_z = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  //transform sigma points into measurement space
  for (int i=0; i < 2*n_aug_+1; i++) {
    VectorXd x_i = Xsig_pred_.col(i);
    // extract values for better readibility
    double px=x_i(0), py=x_i(1);
    // measurement model
    Zsig(0, i) = px;
    Zsig(1, i) = py;
  }

  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i< 2*n_aug_+1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  //calculate measurement covariance matrix S
  MatrixXd R(n_z, n_z);
  R.fill(0.0);
  R(0,0) = std_laspx_ * std_laspx_;
  R(1,1) = std_laspy_ * std_laspy_;
  S.fill(0.0);
  for (int i=0; i< 2*n_aug_+1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  S += R;

  //print result
  // std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  // std::cout << "S: " << std::endl << S << std::endl;

  /*
  ** 2. Update state mean and covariance matrix
  */

  //create z from incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z <<
      meas_package.raw_measurements_[0],   //px
      meas_package.raw_measurements_[1];   //py

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    //state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K
  MatrixXd Si = S.inverse();
  MatrixXd K = Tc * Si;

  //residual
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  //print result
  // std::cout << "Updated state x_: " << std::endl << x_ << std::endl;
  // std::cout << "Updated state covariance P_: " << std::endl << P_ << std::endl;

  //calculate the lidar NIS
  NIS_laser_ = z_diff.transpose() * Si * z_diff;

  // std::cout << "NIS_laser_ = " << NIS_laser_ << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  // std::cout << "---UpdateRadar()---" << std::endl;

  /*
  ** 1. Predict radar measurement
  */
  
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  //transform sigma points into measurement space
  for (int i=0; i < 2*n_aug_+1; i++) {
    VectorXd x_i = Xsig_pred_.col(i);
    // extract values for better readibility
    double px=x_i(0), py=x_i(1), v=x_i(2), psi=x_i(3);
    double rho = sqrt(px*px + py*py);
    double phi = atan2(py, px);
    double rho_dot = (px*cos(psi)*v + py*sin(psi)*v) / rho;
    // measurement model
    Zsig(0, i) = rho;
    Zsig(1, i) = phi;
    Zsig(2, i) = rho_dot;
  }

  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i< 2*n_aug_+1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  //calculate measurement covariance matrix S
  MatrixXd R(n_z, n_z);
  R.fill(0.0);
  R(0,0) = std_radr_ * std_radr_;
  R(1,1) = std_radphi_ * std_radphi_;
  R(2,2) = std_radrd_ * std_radrd_;
  S.fill(0.0);
  for (int i=0; i< 2*n_aug_+1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }
  S += R;

  //print result
  // std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  // std::cout << "S: " << std::endl << S << std::endl;

  /*
  ** 2. Update state mean and covariance matrix
  */

  //create z from incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z <<
      meas_package.raw_measurements_[0],   //rho
      meas_package.raw_measurements_[1],   //phi
      meas_package.raw_measurements_[2];   //rho_dot

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    //state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K
  MatrixXd Si = S.inverse();
  MatrixXd K = Tc * Si;

  //residual
  VectorXd z_diff = z - z_pred;
  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  //print result
  // std::cout << "Updated state x_: " << std::endl << x_ << std::endl;
  // std::cout << "Updated state covariance P_: " << std::endl << P_ << std::endl;

  //calculate the radar NIS
  NIS_radar_ = z_diff.transpose() * Si * z_diff;

  // std::cout << "NIS_radar_ = " << NIS_radar_ << std::endl;

}
