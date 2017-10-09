/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// declare a random engine to be used across multiple and various method calls
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 100;
  
  // Sensor noise at initialization
  normal_distribution<double> noise_init_x(0, std[0]);
  normal_distribution<double> noise_init_y(0, std[1]);
  normal_distribution<double> noise_init_theta(0, std[2]);
  
  // Initialize particles
  for (int i=0; i < num_particles; i++){
    Particle particle;
    particle.id = i;
    particle.x = x;
    particle.y = y;
    particle.theta = theta;
    particle.weight = 1.0;
    
    // Add noise
    particle.x += noise_init_x(gen);
    particle.y += noise_init_y(gen);
    particle.theta += noise_init_theta(gen);
    
    // Put particles in Particles
    particles.push_back(particle);
  }
  
is_initialized = true;
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
  // Add noise at predicition
  normal_distribution<double> noise_pred_x(0, std_pos[0]);
  normal_distribution<double> noise_pred_y(0, std_pos[1]);
  normal_distribution<double> noise_pred_theta(0, std_pos[2]);
  
  for (int i = 0; i < num_particles; i++){
    // calculate new state
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    
    // add noise
    particles[i].x += noise_pred_x(gen);
    particles[i].y += noise_pred_y(gen);
    particles[i].theta += noise_pred_theta(gen);
  }
  
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  
  for (unsigned int i = 0; i < observations.size(); i++) {
    
    // current observation
    LandmarkObs obs = observations[i];
    
    // initiate distance variable with maximum number for double
    double min_dist = numeric_limits<double>::max();
//    double min_dist = 10000000.0;
    
    // landmark id initialization
    int map_id = -1;
    
    for (unsigned int j = 0; j < predicted.size(); j++) {
      // current prediction
      LandmarkObs pred = predicted[j];
      
      // get distance between current/predicted landmarks
      double current_dist = dist(obs.x, obs.y, pred.x, pred.y);
      
      // find nearest landmark to predicted location
      if (current_dist < min_dist) {
        min_dist = current_dist;
        map_id = pred.id;
      }
    }
    
    // set the observation's id to the nearest predicted landmark's id
    observations[i].id = map_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  for (int i = 0; i < num_particles; i++) {
    
    // Get information of a particle
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
    
    // vector for locations in sensor range
    vector<LandmarkObs> predictions;
    
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      
      // get information of a landmark
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      
      // calculate distance
      if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) { // choose this for faster calculation
//       if (dist(lm_x, lm_y, p_x, p_y) <= sensor_range) {                        // choose this for more precise calculation
        
        // add prediction to vector
        predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
      }
    }
    
    // conduct homogeneous transformation on observations
    vector<LandmarkObs> transformed_obs;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double map_x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
      double map_y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;
      transformed_obs.push_back(LandmarkObs{ observations[j].id, map_x, map_y });
    }
    
    // Associate observations and predictions
    dataAssociation(predictions, transformed_obs);
    
    // re-init weight
    particles[i].weight = 1.0;
    
    for (unsigned int j = 0; j < transformed_obs.size(); j++) {
      
      // set variables for observed x, y and predicted x, y
      double obs_x, obs_y, pr_x, pr_y;
      obs_x = transformed_obs[j].x;
      obs_y = transformed_obs[j].y;
      
      int associated_prediction = transformed_obs[j].id;
      
      // find matched id
      for (unsigned int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == associated_prediction) {
          pr_x = predictions[k].x;
          pr_y = predictions[k].y;
        }
      }
      
      // determine measurement probabilities
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double obs_w = ( 1/(2*M_PI*std_x*std_y)) * exp( -( pow(pr_x-obs_x,2)/(2*pow(std_x, 2)) + (pow(pr_y-obs_y,2)/(2*pow(std_y, 2))) ) );
      
      // combine probabilities
      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  vector<Particle> new_particles;
  
  // get all of the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }
  
  // generate random index using discrete_distribution
  discrete_distribution<int> discrete_int_dist(0, num_particles-1);
  auto index = discrete_int_dist(gen);
  
  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());
  
  // generate random uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> uni_real_dist(0.0, max_weight);
  
  double beta = 0.0;
  
  // resampling wheel
  for (int i = 0; i < num_particles; i++) {
    beta += uni_real_dist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
