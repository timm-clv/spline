#pragma once
#include "SimpleNN.h"
#include <array>
#include <vector>

struct Point { float x, y; };
struct RobotState { float x, y, yaw, v; };

class VelocityController {
private:
    SimpleNN nn_model;
    const float V_MAX = 3.5f;
    const float A_MAX = 4.0f;

    // Utilitaires Math
    void get_bezier_derivatives(float t, Point P0, Point P1, Point P2, Point P3, float T, float& v_norm, float& a_tan);
    
public:
    VelocityController();
    bool init(const std::string& model_path);
    
    // Fonction principale
    // Retourne le temps optimal T et remplit les points de contrôle
    float compute_trajectory(RobotState robot, RobotState target, std::array<Point, 4>& out_ctrls);
};