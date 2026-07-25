#include "standard.hpp"
#include <iostream>
#include "../../include/logger.hpp"

// SceneStatus::SceneStatus() : GoldScene(false), CarScene(false), HumanScene(false), LightScene(false) {}

// bool SceneStatus::all() const
// {
//     return GoldScene || CarScene || HumanScene || LightScene;
// }

// struct SceneStatus//????????????
// {
//     bool GoldScene;
//     bool CarScene;
//     bool HumanScene;
//     bool LightScene;

//     SceneStatus()
//     {
//         this->GoldScene = false;
//         this->CarScene = false;
//         this->HumanScene = false;
//         this->LightScene = false;
//     }

//     bool all()
//     {
//         return this->GoldScene || this->CarScene || this->HumanScene || this->LightScene;
//     }
// } scene_status;

Standard::Standard(const Config *config)
    : trackstate(TrackState::TRACK_MIDDLE),
      scene_status(Scene_status::Normal_status),
      config_(config),
      gold_(config),
      car_(config),
      human_(config),
      light_(config)
{
    if (config_ == nullptr)
    {
        std::string err;
        if (read_json_bool("./configs/config.json", "enAI", fallback_config_.enAI, err) ||
            read_json_bool("./configs/config.json", "mode.enAI", fallback_config_.enAI, err))
        {
            config_ = &fallback_config_;
            std::cout << "[Standard] config loaded from ./configs/config.json, enAI="
                      << (config_->enAI ? "true" : "false") << "\n";
        }
        else
        {
            std::cerr << "[Standard] config load failed: " << err
                      << ", use default enAI=false\n";
        }
    }

    gold_.GOLD_DIST_FORWARD = config_.GOLD_DIST_FORWARD;
    gold_.GOLD_DIST_BACKWARD = config_.GOLD_DIST_BACKWARD;
    car_.CAR_DIST_FORWARD = config_.CAR_DIST_FORWARD;
    car_.CAR_DIST_BACKWARD = config_.CAR_DIST_BACKWARD;
    human_.HUMAN_DIST_FORWARD = config_.HUMAN_DIST_FORWARD;
    human_.HUMAN_DIST_BACKWARD = config_.HUMAN_DIST_BACKWARD;

    gold_.GOLD_FIND_FRAMES_THRESH = config_.GOLD_FIND_FRAMES_THRESH;
    gold_.GOLD_LOST_FRAMES_THRESH = config_.GOLD_LOST_FRAMES_THRESH;
    car_.CAR_FIND_FRAMES_THRESH = config_.CAR_FIND_FRAMES_THRESH;
    car_.CAR_LOST_FRAMES_THRESH = config_.CAR_LOST_FRAMES_THRESH;
    human_.HUMAN_FIND_FRAMES_THRESH = config_.HUMAN_FIND_FRAMES_THRESH;
    human_.HUMAN_LOST_FRAMES_THRESH = config_.HUMAN_LOST_FRAMES_THRESH;

    gold_.confidence_threshold = config_.confidence_threshold;
    car_.confidence_threshold = config_.confidence_threshold;
    human_.confidence_threshold = config_.confidence_threshold;
    light_.confidence_threshold = config_.confidence_threshold;
}

TaskData Standard::run(const cv::Mat &src_img,
                       const std::vector<PredictResult> &predict_result,
                       double pitch_angle)
{

    cv::Mat imgShow = src_img.clone();

    (void)pitch_angle;
    int obj_count = 0;
    (void)obj_count;
    /********************************** 图像处理 ***********************************/

    /******************************* AI检测状态更新 ********************************/

    updateSceneStatus(frame, scene_status, config_);
    const bool enAI = (config_ != nullptr) ? config_->enAI : false;

    /********************************** 元素判断 ***********************************/

    if (scene_status == Scene_status::Normal && enAI)
    {
        if (gold_.flag_gold == GoldFlag::GOLD_NONE)
        {
            gold_.check_gold(predict_result);
            if (gold_.flag_gold != GoldFlag::GOLD_NONE)
            {
                scene_status = Scene_status::Gold_status;
            }
        }

        if (car_.flag_car == CarFlag::CAR_NONE)
        {
            car_.check_car(predict_result);
            if (car_.flag_car != CarFlag::CAR_NONE)
            {
                scene_status = Scene_status::Car_status;
            }
        }

        if (human_.human_flag == HumanFlag::HUMAN_NONE)
        {
            human_.check_human(predict_result);
            if (human_.human_flag != HumanFlag::HUMAN_NONE)
            {
                scene_status = Scene_status::Human_status;
            }
        }

        if (light_.flag_light == LightFlag::LIGHT_NONE)
        {
            light_.check_light(predict_result);
            if (light_.flag_light != LightFlag::LIGHT_NONE)
            {
                scene_status = Scene_status::Light_status;
            }
        }
        // else if (light_.flag_light == LightFlag::LIGHT_LOST)
        // {
        //     // Placeholder for lost logic.
        // }
    }

    if (scene_status == Scene_status::Light_status)
    {
        light_.run_light(src_img, predict_result, CV_pointsOrigin, trackPoints, CV_pointsOrigin_size, trackPoints_size);
        logger("LightScene");
        if (light_.flag_light == LightFlag::LIGHT_NONE)
        {
            scene_status = Scene_status::Normal_status;
        }
    }
    else if (scene_status == Scene_status::Gold_status)
    {
        gold_.run_gold(src_img, predict_result, CV_pointsOrigin, trackPoints, CV_pointsOrigin_size, trackPoints_size);
        logger("GoldScene");
        if (gold_.flag_gold == GoldFlag::GOLD_NONE)
        {
            scene_status = Scene_status::Normal_status;
        }
    }
    else if (scene_status == Scene_status::Car_status)
    {
        car_.run_car(src_img, predict_result, CV_pointsOrigin, trackPoints, CV_pointsOrigin_size, trackPoints_size);
        logger("CarScene");
        if (car_.flag_car == CarFlag::CAR_NONE)
        {
            scene_status = Scene_status::Normal_status;
        }
    }
    else if (scene_status == Scene_status::Human_status)
    {
        human_.run_human(src_img, predict_result, CV_pointsOrigin, trackPoints, CV_pointsOrigin_size, trackPoints_size);
        logger("HumanScene");
        if (human_.human_flag == HumanFlag::HUMAN_NONE)
        {
            scene_status = Scene_status::Normal_status;
        }
    }

    /********************************** cv传统巡线 ***********************************/

    std::vector<cv::Point> lane_points = tracker.ExtractArrows(frame, mask);

    if (lane_points.size() > 3)
    {
        trackPoints_CV.clear();
        trackPoints_CV = tracker.FitTrajectory_LOWESS(PTS_LINE_NUM, lane_points, frame);
        trackPoints_CV_size = trackPoints_CV.size();

        if (trackstate == TrackState::TRACK_MIDDLE)
        {
            LineTracker::ExtractArrows(src_img, debug_mask);
        }
        else if (trackstate == TrackState::TRACK_AI_MIDDLE)
        {
            LineTracker::ExtractArrows(src_img, debug_mask);
        }

        /********************************** 透视变换 ***********************************/
        t_trackPoints_CV.clear();
        for (int i = 0; i < trackPoints_CV.size(); i++)
        {
            cv::Point2f t_pt = general.transf(trackPoints_CV[i].x, trackPoints_CV[i].y);
            if ((t_pt.x < 640 - 1 && t_pt.x > 0) && (t_pt.y < 480 - 1 && t_pt.y > 0))
            {
                t_trackPoints_CV.emplace_back(cvRound(t_pt.x), cvRound(t_pt.y));
            }
        }
        int t_trackPoints_CV_size = t_trackPoints_CV.size();

        /********************************** 起始点重采样 ***********************************/

        if (t_trackPoints_CV.size() > 3)
        {
            // 初始化 s_t_trackPoints_CV 从 trackPoints_CV
            s_t_trackPoints_CV.clear();
            for (const auto &pt : t_trackPoints_CV)
            {
                s_t_trackPoints_CV.emplace_back(pt.x, pt.y);
            }
            s_t_trackPoints_CV_size = s_t_trackPoints_CV.size();

            float min_dist = 10000000;
            int begin_id = -1;
            bool center_effective_flag = false; // 中线有效标志

            cv::Point2f car_base_ipm = general.transf(COLSIMAGE / 2.0f, ROWSIMAGE * 0.95f);
            float cx = car_base_ipm.x;
            float cy = car_base_ipm.y;

            // 找最近点(起始点中线归一化)
            for (int i = 0; i < s_t_trackPoints_CV_size; i++)
            {
                float dx = s_t_trackPoints_CV[i].x - cx;
                float dy = s_t_trackPoints_CV[i].y - cy;
                float dist = sqrt(dx * dx + dy * dy);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    begin_id = i;
                }
            }

            begin_id = general.clip(begin_id, 0, s_t_trackPoints_CV_size - 1);

            /********************************** 等距采样 ***********************************/

            std::vector<POINT> temp_center;
            int temp_center_size;
            if ((begin_id >= 0 && s_t_trackPoints_CV_size - begin_id >= 3))
            {
                center_effective_flag = true;
                if (begin_id >= 0)
                {
                    cx = s_t_trackPoints_CV[begin_id].x;
                    cy = s_t_trackPoints_CV[begin_id].y;
                }
                for (int i = begin_id; i < s_t_trackPoints_CV_size; i++)
                {
                    temp_center.emplace_back(s_t_trackPoints_CV[i].x, s_t_trackPoints_CV[i].y);
                }

                temp_center_size = temp_center.size();
                s_t_trackPoints_CV.clear();
                s_t_trackPoints_CV_size = 0;
                resample_points(temp_center, temp_center_size, s_t_trackPoints_CV,
                                s_t_trackPoints_CV_size, SAMPLE_DIST * pixel_per_meter);

                /********************************** 偏差计算 ***********************************/
                double min_dis = 1000000;

                for (int i = 1; i < s_t_trackPoints_CV_size; i++)
                {
                    double dx = s_t_trackPoints_CV[i].x - cx;
                    double dy = cy - s_t_trackPoints_CV[i].y;
                    double dn = sqrt(dx * dx + dy * dy);

                    double dis = aim_distance_f * pixel_per_meter - dn;
                    if (dis < 0)
                    {
                        dis *= -1;
                    }
                    if (dis < min_dis)
                    {
                        aim_index_far = i;
                        min_dis = dis;
                    }
                }

                float dx = s_t_trackPoints_CV[aim_index_far].x - cx; // rptsn[aim_idx__far][0] - cx;
                float dy = cy - s_t_trackPoints_CV[aim_index_far].y; // cy - rptsn[aim_idx__far][1];
                float dn = sqrt(dx * dx + dy * dy);
                // float error = (-atanf(pixel_per_meter * 2 * car_length * dx / dn / dn) * 180 / PI);
                float error = -atan2f(dx, dy) * 180 / PI;

                assert(!isnan(error));
                printf("far_dx:%f,far_dy %f\n", dx, dy);
                printf("cx %f cy %f\n", cx, cy);
                printf("error_far: %f degrees\n", error); // 添加打印偏差角

                /********************************** 人工势场法偏差计算 ***********************************/

                // 源于蓝线预瞄点的引导力，保证车宏观上始终沿着赛道开
                float base_dx = s_t_trackPoints_CV[aim_index_far].x - cx;
                float base_dy = cy - s_t_trackPoints_CV[aim_index_far].y;

                // 初始化合成力向量 (初始化为基础路径引力，基础引力权重隐式设为1.0)
                float Fx_total = base_dx * 1.0;
                float Fy_total = base_dy * 1.0;

                // --- APF 势场参数 ---
                const float K_ATT_GOLD = 0.5f;     // 金币引力增益
                const float K_REP_OBS = 120000.0f; // 障碍物斥力增益
                const float REP_RADIUS = 280.0f;   // 斥力影响半径

                for (const auto &obj : predict_result)
                {
                    float obj_cx = obj.x + obj.w / 2.0f;
                    float obj_cy = obj.y + obj.h;

                    float dx_obj = obj_cx - cx;
                    float dy_obj = cy - obj_cy;
                    float dist = sqrt(dx_obj * dx_obj + dy_obj * dy_obj);

                    if (dist < 1e-2)
                        continue; // 防止除零异常

                    // 金币附加引力
                    if (obj.class_id == Standard::CLASS_ID_GOLD)
                    {
                        // 使用线性引力模型
                        Fx_total += K_ATT_GOLD * dx_obj;
                        Fy_total += K_ATT_GOLD * dy_obj;
                    }
                    // 车、行人障碍物斥力
                    else if (obj.class_id == Standard::CLASS_ID_CAR || obj.class_id == Standard::CLASS_ID_HUMAN)
                    {
                        if (dist < REP_RADIUS)
                        {
                            // 斥力大小计算F = K * (1/d - 1/R) / d^2
                            float rep_mag = K_REP_OBS * (1.0f / dist - 1.0f / REP_RADIUS) / (dist * dist);

                            Fx_total += rep_mag * (-dx_obj / dist);
                            Fy_total += rep_mag * (-dy_obj / dist);
                        }
                    }
                }

                // 动力学映射：使用总合成力计算最终的控制偏差角
                float error = -atan2f(Fx_total, Fy_total) * 180 / PI;

                assert(!isnan(error));
                printf("APF_Fx: %f, APF_Fy: %f\n", Fx_total, Fy_total);
                printf("cx %f cy %f\n", cx, cy);
                printf("error_far: %f degrees\n", error); // 打印最新的偏差角

                /***************************** 绘图 ********************************/
                cv::Mat imgT;

                if (_config.en_show || _config.saveImg)
                {
                    warpPerspective(src_img, imgT, general.rotation, src_img.size());
                    cv::putText(imgT, "error" + std::to_string(aim_angle_filter), Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255));

                    // filtered_line_CV绘图
                    for (size_t i = 0; i + 1 < trackPoints_CV.size(); ++i)
                    {
                        cv::circle(out_img, trackPoints_CV[i], 4, cv::Scalar(0, 255, 255), -1);
                    }
                    for (size_t i = 0; i + 1 < filtered_line_CV.size(); ++i)
                    {
                        cv::line(out_img, filtered_line_CV[i], filtered_line_CV[i + 1], cv::Scalar(255, 0, 0), 3);
                    }
                }

                if (_config.saveImg)
                {
                    general.savePicture(imgT);
                    // general.savePicture(src_img);
                    // general.savePicture(imgShow);
                    // videoWriter << imgT;
                }
            }
        }
    }
}

void updateSceneStatus(const aiget::PredictFrame &frame, Scene_status &scene_status, const Config *config)
{

    float confidence_threshold = (config != nullptr) ? config->confidence_threshold : 0.5f;

    // 重置所有标志
    scene_status.GoldScene = false;
    scene_status.CarScene = false;
    scene_status.HumanScene = false;
    scene_status.LightScene = false;

    // 临时 map 用于存储每种 class_id 的候选对象（选择 y2 最大的）
    std::map<int, aiget::PredictResult> candidates;

    for (const auto &obj : frame.objects)
    {
        if (obj.score > confidence_threshold)
        {
            int cid = obj.class_id;
            // 检查是否已有候选，或当前对象的 y2 更大
            if (candidates.find(cid) == candidates.end() || obj.y2 > candidates[cid].y2)
            {
                candidates[cid] = obj;
            }
            // 更新 Scene_status 标志（原有逻辑）
            switch (cid)
            {
            case Standard::CLASS_ID_GOLD:
                scene_status.GoldScene = true;
                break;
            case Standard::CLASS_ID_CAR:
                scene_status.CarScene = true;
                break;
            case Standard::CLASS_ID_HUMAN:
                scene_status.HumanScene = true;
                break;
            case Standard::CLASS_ID_LIGHT:
                scene_status.LightScene = true;
                break;
            default:
                break;
            }
        }
    }

    // 清空上一帧的数据
    std_instance.gold_results_.clear();
    std_instance.car_results_.clear();
    std_instance.human_results_.clear();
    std_instance.light_results_.clear();

    t_gold_points.clear();
    t_human_points.clear();

    // 将选中的候选对象写入对应全局数组
    for (const auto &pair : candidates)
    {
        int cid = pair.first;
        const auto &selected_obj = pair.second;
        switch (cid)
        {
        case Standard::CLASS_ID_GOLD:
            std_instance.gold_results_.push_back(selected_obj);
            gold_point.x = (selected_obj.x1 + selected_obj.x2) / 2;
            gold_point.y = (selected_obj.y1 + selected_obj.y2) / 2;
            t_gold_point = general.transf(gold_point.x, gold_point.y);
            break;
        case Standard::CLASS_ID_CAR:
            std_instance.car_results_.push_back(selected_obj);
            car_point.x = (selected_obj.x1 + selected_obj.x2) / 2;
            car_point.y = (selected_obj.y1 + selected_obj.y2) / 2;
            t_car_point = general.transf(car_point.x, car_point.y);
            break;
        case Standard::CLASS_ID_HUMAN:
            std_instance.human_results_.push_back(selected_obj);
            human_point.x = (selected_obj.x1 + selected_obj.x2) / 2;
            human_point.y = (selected_obj.y1 + selected_obj.y2) / 2;
            t_human_point = general.transf(human_point.x, human_point.y);
            break;
        case Standard::CLASS_ID_LIGHT:
            std_instance.light_results_.push_back(selected_obj);
            light_point.x = (selected_obj.x1 + selected_obj.x2) / 2;
            light_point.y = (selected_obj.y1 + selected_obj.y2) / 2;
            t_light_point = general.transf(light_point.x, light_point.y);
            break;
        }
    }
}

void Standard::trackRecognition(cv::Mat &imageGray, cv::Mat &imageBinary, cv::Mat &imgShow)
{
    trackPoints_CV.clear();
    filtered_line_CV.clear();
    t_trackPoints_CV.clear();
    s_t_trackPoints_CV.clear();
    t_angle_CV.clear();

    s_t_trackPoints_APF.clear();

    trackPoints_AI.clear();
    t_trackPoints_AI.clear();
    s_t_trackPoints_AI.clear();
    t_angle_AI.clear();
    t_CenterEdge.clear();

    trackPoints_CV_size = 0;
    filtered_line_CV_size = 0;
    t_trackPoints_CV_size = 0;
    s_t_trackPoints_CV_size = 0;
    t_angle_CV_size = 0;

    s_t_trackPoints_APF_size = 0;

    trackPoints_AI_size = 0;
    t_trackPoints_AI_size = 0;
    s_t_trackPoints_AI_size = 0;
    t_angle_AI_size = 0;
    t_CenterEdge_size = 0;

    if (trackstate == TrackState::TRACK_MIDDLE)
    {
        trackPoints_CV = tracker.ExtractArrows(frame, mask);
    }
    else if (trackstate == TrackState::TRACK_AI_MIDDLE)
    {
        trackPoints_AI = tracker.ExtractArrows(frame, mask);
    }

    // 中线拟合
    filtered_line_CV = FitTrajectory_Poly((int)trackPoints_CV.size(), trackPoints_CV, frame);
}

/**
 * @brief 等距采样
 */
void resample_points(const vector<POINT> &input, int input_size, vector<POINT> &output, int &output_size, float dist_threshold)
{
    output.clear();

    //  0.02 * 200 = 4.0
    if (dist_threshold < 2.0f)
    {
        dist_threshold = 4.0f;
    }

    if (input_size < 2)
    {
        output = input;
        output_size = output.size();
        return;
    }

    output.push_back(input[0]);
    float current_dist = 0.0f;
    float target_dist = dist_threshold;

    for (int i = 1; i < input_size; ++i)
    {
        float dx = input[i].x - input[i - 1].x;
        float dy = input[i].y - input[i - 1].y;
        float segment_len = std::sqrt(dx * dx + dy * dy);

        while (current_dist + segment_len >= target_dist)
        {
            float ratio = (target_dist - current_dist) / segment_len;

            float nx = input[i - 1].x + dx * ratio;
            float ny = input[i - 1].y + dy * ratio;

            output.push_back(cv::Point(cvRound(nx), cvRound(ny)));

            target_dist += dist_threshold;
        }
        current_dist += segment_len;
    }
    output_size = output.size();
}

/**
 * @brief 动态预瞄点计算
 */
double Standard::DynamicAimDisCal()
{
    vector<PointsCurve> CurveEdge;
    vector<Point2d> d_s_t_trackPoints_CV;
    for (int i = 0; i < s_t_trackPoints_CV_size; i++)
    {
        d_s_t_trackPoints_CV.emplace_back(Point2d(s_t_trackPoints_CV[i].x, s_t_trackPoints_CV[i].y));
    }
    vector<double> t(d_s_t_trackPoints_CV.size());
    for (size_t i = 0; i < d_s_t_trackPoints_CV.size(); i++)
    {
        t[i] = i; // 参数化
    }
    for (int i = 1; i < s_t_trackPoints_CV_size - 1; i++)
    {
        PointsCurve sp;
        sp.x = d_s_t_trackPoints_CV[i].x;
        sp.y = d_s_t_trackPoints_CV[i].y;

        // 使用中心差分计算一阶导数
        double dt1 = t[i] - t[i - 1];
        double dt2 = t[i + 1] - t[i];

        sp.dx = ((d_s_t_trackPoints_CV[i].x - d_s_t_trackPoints_CV[i - 1].x) / dt1 +
                 (d_s_t_trackPoints_CV[i + 1].x - d_s_t_trackPoints_CV[i].x) / dt2) /
                2.0;
        sp.dy = ((d_s_t_trackPoints_CV[i].y - d_s_t_trackPoints_CV[i - 1].y) / dt1 +
                 (d_s_t_trackPoints_CV[i + 1].y - d_s_t_trackPoints_CV[i].y) / dt2) /
                2.0;

        // 计算二阶导数
        sp.ddx = (d_s_t_trackPoints_CV[i + 1].x - 2 * d_s_t_trackPoints_CV[i].x + d_s_t_trackPoints_CV[i - 1].x);
        sp.ddy = (d_s_t_trackPoints_CV[i + 1].y - 2 * d_s_t_trackPoints_CV[i].y + d_s_t_trackPoints_CV[i - 1].y);

        // 计算曲率 κ = |x'y'' - y'x''| / (x'? + y'?)^(3/2)
        double numerator = abs(sp.dx * sp.ddy - sp.dy * sp.ddx);
        double denominator = pow(sp.dx * sp.dx + sp.dy * sp.dy, 1.5);

        sp.curve = (denominator > 1e-10) ? numerator / denominator : 0.0;

        CurveEdge.push_back(sp);
    }
    if (!CurveEdge.empty())
    {
        double sum = 0.0;
        double maxCurvature = 0.0;
        double maxCurvatureIndex = 0;

        for (size_t i = 0; i < CurveEdge.size(); i++)
        {
            sum += CurveEdge[i].curve;
            if (CurveEdge[i].curve > maxCurvature)
            {
                maxCurvature = CurveEdge[i].curve;
                maxCurvatureIndex = i;
            }
        }

        double meanCurvature = sum / CurveEdge.size();
        printf("最大曲率 %f n\n", maxCurvature);
    }
    return 0.0;
}