import math
import numpy as np

class TrajectoryFitter:
    def fit_trajectory_lsm(self, pts, frame):
        """最小二乘法二次项拟合: x = a*y^2 + b*y + c"""
        curve_pts =[]
        if len(pts) < 3: return curve_pts

        y_coords = np.array([p[1] for p in pts])
        x_coords = np.array([p[0] for p in pts])

        coeffs = np.polyfit(y_coords, x_coords, 2)

        y_start = pts[0][1]
        y_end = pts[-1][1]

        if y_start <= y_end: return curve_pts

        for y in range(y_start, y_end - 1, -5):
            x = int(round(np.polyval(coeffs, y)))
            if 0 <= x < frame.shape[1]:
                curve_pts.append((x, y))
                
        return curve_pts

    def fit_trajectory_lowess(self, point_count, pts, frame):
        """LOWESS 局部加权回归拟合"""
        n = len(pts)
        if n < 2: return pts
        curve_pts =[]
        span = 0.4
        window_size = max(3, int(n * span))

        for i in range(point_count):
            t = i / (point_count - 1) * (n - 1)
            sum_w = sum_wt = sum_wt2 = sum_wx = sum_wy = sum_wtx = sum_wty = 0.0

            for j in range(n):
                diff = abs(t - j)
                if diff < window_size:
                    rel_dist = diff / window_size
                    w = math.pow(1.0 - math.pow(rel_dist, 3), 3)

                    sum_w += w
                    sum_wt += w * j
                    sum_wt2 += w * j * j
                    sum_wx += w * pts[j][0]
                    sum_wy += w * pts[j][1]
                    sum_wtx += w * j * pts[j][0]
                    sum_wty += w * j * pts[j][1]

            det = sum_w * sum_wt2 - sum_wt * sum_wt
            if abs(det) > 1e-6:
                x_fit = (sum_wt2 * sum_wx - sum_wt * sum_wtx) / det + (sum_w * sum_wtx - sum_wt * sum_wx) / det * t
                y_fit = (sum_wt2 * sum_wy - sum_wt * sum_wty) / det + (sum_w * sum_wty - sum_wt * sum_wy) / det * t
                curve_pts.append((int(round(x_fit)), int(round(y_fit))))
                
        return curve_pts

    def fit_trajectory_poly(self, point_count, pts, frame):
        """参数方程多项式拟合"""
        n = len(pts)
        degree = 3
        if n <= degree: return pts

        x_coords = [p[0] for p in pts]
        y_coords = [p[1] for p in pts]
        t_coords = [i / (n - 1) for i in range(n)]

        coeff_x = np.polyfit(t_coords, x_coords, degree)
        coeff_y = np.polyfit(t_coords, y_coords, degree)

        curve_pts =[]
        for i in range(point_count):
            t = i / (point_count - 1)
            rx = np.polyval(coeff_x, t)
            ry = np.polyval(coeff_y, t)
            curve_pts.append((int(round(rx)), int(round(ry))))
            
        return curve_pts

    def fit_trajectory_gpr(self, point_count, pts, frame):
        """高斯过程回归"""
        n = len(pts)
        if n < 2: return pts

        l = 0.6
        sigma_f = 1.0
        sigma_n = 0.1

        def kernel(t1, t2):
            return sigma_f * math.exp(-((t1 - t2) ** 2) / (2 * l * l))

        K = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                t_i = i / (n - 1)
                t_j = j / (n - 1)
                K[i, j] = kernel(t_i, t_j) + (sigma_n ** 2 if i == j else 0)

        K_inv = np.linalg.inv(K)

        Y_x = np.array([p[0] for p in pts], dtype=np.float64).reshape(-1, 1)
        Y_y = np.array([p[1] for p in pts], dtype=np.float64).reshape(-1, 1)

        curve_pts =[]
        for i in range(point_count):
            t_star = i / (point_count - 1)
            k_star = np.array([kernel(t_star, j / (n - 1)) for j in range(n)]).reshape(1, -1)

            mu_x = k_star @ K_inv @ Y_x
            mu_y = k_star @ K_inv @ Y_y

            curve_pts.append((int(round(float(mu_x[0, 0]))), int(round(float(mu_y[0, 0])))))
            
        return curve_pts

    def fit_trajectory_bezier(self, point_count, lane_points):
        """分段贝塞尔曲线"""
        if point_count < 2 or not lane_points or len(lane_points) != point_count:
            return lane_points

        dt = 0.02
        output =[]

        if point_count == 2:
            for t in np.arange(0, 1 + 1e-6, dt):
                x = int(round((1 - t) * lane_points[0][0] + t * lane_points[1][0]))
                y = int(round((1 - t) * lane_points[0][1] + t * lane_points[1][1]))
                output.append((x, y))
            return output

        if point_count == 3:
            for t in np.arange(0, 1 + 1e-6, dt):
                t1 = 1 - t
                x = int(round(t1*t1*lane_points[0][0] + 2*t1*t*lane_points[1][0] + t*t*lane_points[2][0]))
                y = int(round(t1*t1*lane_points[0][1] + 2*t1*t*lane_points[1][1] + t*t*lane_points[2][1]))
                output.append((x, y))
            return output

        def calc_cubic_bezier(ctrl_pts):
            segment =[]
            for t in np.arange(0, 1 + 1e-6, dt):
                t1 = 1 - t
                t1_3, t1_2, t_2, t_3 = t1**3, t1**2, t**2, t**3
                x = int(round(t1_3*ctrl_pts[0][0] + 3*t1_2*t*ctrl_pts[1][0] + 3*t1*t_2*ctrl_pts[2][0] + t_3*ctrl_pts[3][0]))
                y = int(round(t1_3*ctrl_pts[0][1] + 3*t1_2*t*ctrl_pts[1][1] + 3*t1*t_2*ctrl_pts[2][1] + t_3*ctrl_pts[3][1]))
                segment.append((x, y))
            return segment

        # 每4个控制点为一段，重叠1个控制点
        for i in range(0, len(lane_points) - 3, 3):
            ctrl_pts = lane_points[i:i+4]
            output.extend(calc_cubic_bezier(ctrl_pts))

        # 处理最后不足4个点的剩余部分
        if len(lane_points) % 3 != 1 and len(lane_points) > 4:
            last_ctrl_pts = lane_points[-4:]
            output.extend(calc_cubic_bezier(last_ctrl_pts))

        return output