#include "standard.h"

#include <iostream>

#include "../include/config_loader.h"

SceneStatus::SceneStatus() : GoldScene(false), CarScene(false), HumanScene(false), LightScene(false) {}

bool SceneStatus::all() const {
    return GoldScene || CarScene || HumanScene || LightScene;
}

Standard::Standard(const Config* config)
    : trackstate(TrackState::TRACK_MIDDLE),
      scene(Scene::Normal),
      config_(config),
      fallback_config_{},
      gold_end_counter_(0),
      car_end_counter_(0),
      human_end_counter_(0),
      gold_(config),
      car_(config),
      human_(config),
      light_(config) {
    if (config_ == nullptr) {
        std::string err;
        if (read_json_bool("./configs/config.json", "enAI", fallback_config_.enAI, err) ||
            read_json_bool("./configs/config.json", "mode.enAI", fallback_config_.enAI, err)) {
            config_ = &fallback_config_;
            std::cout << "[Standard] config loaded from ./configs/config.json, enAI="
                      << (config_->enAI ? "true" : "false") << "\n";
        } else {
            std::cerr << "[Standard] config load failed: " << err
                      << ", use default enAI=false\n";
        }
    }
}

TaskData Standard::run(const cv::Mat& src_img,
                       const std::vector<DetectionTarget>& predict_result,
                       double pitch_angle) {
    (void)pitch_angle;
    int same_points = 0;
    (void)same_points;

    if (trackstate == TrackState::TRACK_MIDDLE) {
        // Keep placeholder branch to match Python behavior.
    } else if (trackstate == TrackState::TRACK_AI_MIDDLE) {
        // Keep placeholder branch to match Python behavior.
    }

    const bool enAI = (config_ != nullptr) ? config_->enAI : false;
    if (scene == Scene::Normal && enAI) {
        if (gold_.flag_gold == GoldFlag::GOLD_NONE) {
            gold_.check_gold(predict_result);
            if (gold_.flag_gold != GoldFlag::GOLD_NONE) {
                // Placeholder for scene switch.
            }
        }

        if (car_.flag_car == CarFlag::CAR_NONE) {
            car_.check_car(predict_result);
            if (car_.flag_car != CarFlag::CAR_NONE) {
                // Placeholder for scene switch.
            }
        }

        if (human_.flag_human == HumanFlag::HUMAN_NONE) {
            human_.check_human(predict_result);
            if (human_.flag_human != HumanFlag::HUMAN_NONE) {
                // Placeholder for scene switch.
            }
        }

        if (light_.flag_light == LightFlag::LIGHT_NONE) {
            light_.check_light(predict_result);
            if (light_.flag_light != LightFlag::LIGHT_NONE) {
                // Placeholder for scene switch.
            }
        } else if (light_.flag_light == LightFlag::LIGHT_LOST) {
            // Placeholder for lost logic.
        }
    }

    ++gold_end_counter_;
    if (gold_end_counter_ > 600) {
        gold_end_counter_ = 599;
    }

    if (scene == Scene::Gold) {
        gold_.run_gold(src_img.empty() ? nullptr : &src_img, predict_result);
        if (gold_.flag_gold == GoldFlag::GOLD_NONE) {
            // Placeholder for leaving gold scene.
        }
    } else if (scene == Scene::Car) {
        // Placeholder for car scene handler.
    } else {
        // Placeholder for default scene handler.
    }

/**
 * @brief 动态预瞄点计算
 */
double Standard::DynamicAimDisCal()
{
    vector<PointsCurve> CurveEdge;
    vector<Point2d> d_t_CenterEdge;
    for(int i = 0;i< t_CenterEdge_size;i++)
    {
        d_t_CenterEdge.emplace_back(Point2d(t_CenterEdge[i].x,t_CenterEdge[i].y));
    }
    vector<double> t(d_t_CenterEdge.size());
    for (size_t i = 0; i < d_t_CenterEdge.size(); i++) {
        t[i] = i; // 参数化
    }
    for(int i = 1;i < t_CenterEdge_size - 1; i ++)
    {
        PointsCurve sp;
        sp.x = d_t_CenterEdge[i].x;
        sp.y = d_t_CenterEdge[i].y;
        
        // 使用中心差分计算一阶导数
        double dt1 = t[i] - t[i-1];
        double dt2 = t[i+1] - t[i];
        
        sp.dx = ((d_t_CenterEdge[i].x - d_t_CenterEdge[i-1].x) / dt1 + 
                    (d_t_CenterEdge[i+1].x - d_t_CenterEdge[i].x) / dt2) / 2.0;
        sp.dy = ((d_t_CenterEdge[i].y - d_t_CenterEdge[i-1].y) / dt1 + 
                    (d_t_CenterEdge[i+1].y - d_t_CenterEdge[i].y) / dt2) / 2.0;
        
        // 计算二阶导数
        sp.ddx = (d_t_CenterEdge[i+1].x - 2*d_t_CenterEdge[i].x + d_t_CenterEdge[i-1].x);
        sp.ddy = (d_t_CenterEdge[i+1].y - 2*d_t_CenterEdge[i].y + d_t_CenterEdge[i-1].y);
        
        // 计算曲率 κ = |x'y'' - y'x''| / (x'? + y'?)^(3/2)
        double numerator = abs(sp.dx * sp.ddy - sp.dy * sp.ddx);
        double denominator = pow(sp.dx * sp.dx + sp.dy * sp.dy, 1.5);
        
        sp.curve = (denominator > 1e-10) ? numerator / denominator : 0.0;
        
        CurveEdge.push_back(sp);   
    }
    if (!CurveEdge.empty()) {
        double sum = 0.0;
        double maxCurvature = 0.0;
        double maxCurvatureIndex = 0;
        
        for (size_t i = 0; i < CurveEdge.size(); i++) {
            sum += CurveEdge[i].curve;
            if (CurveEdge[i].curve > maxCurvature) {
                maxCurvature = CurveEdge[i].curve;
                maxCurvatureIndex = i;
            }
        }
        
        double meanCurvature = sum / CurveEdge.size();
        printf("最大曲率 %f n\n",maxCurvature);
        
    }
    return 0.0;


    return TaskData{};
}