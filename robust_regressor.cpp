#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <numeric>
#include <iomanip>
#include <chrono>

// ==================== 数据结构 ====================

struct Point2D {
    double x, y;
    int index;
};

struct Point3D {
    double x, y, z;
    int index;
};

struct FitResult {
    std::vector<double> coeffs;       // 多项式/平面系数
    std::vector<int> inlier_indices;  // 正常点索引
    std::vector<int> outlier_indices; // 异常点索引
    double rmse;
    double r_squared;
    int model_degree;
};

// ==================== 数学工具 ====================

class MathUtils {
public:
    // 高斯消元（部分主元）解 Ax = b
    static std::vector<double> solveLinearSystem(std::vector<std::vector<double>> A, std::vector<double> b) {
        int n = b.size();
        for (int i = 0; i < n; ++i) {
            double maxVal = std::abs(A[i][i]);
            int maxRow = i;
            for (int k = i + 1; k < n; ++k) {
                if (std::abs(A[k][i]) > maxVal) {
                    maxVal = std::abs(A[k][i]);
                    maxRow = k;
                }
            }
            for (int k = i; k < n; ++k) std::swap(A[maxRow][k], A[i][k]);
            std::swap(b[maxRow], b[i]);

            if (std::abs(A[i][i]) < 1e-12) {
                return std::vector<double>(n, 0.0);
            }

            for (int k = i + 1; k < n; ++k) {
                double c = -A[k][i] / A[i][i];
                for (int j = i; j < n; ++j) {
                    if (i == j) A[k][j] = 0;
                    else A[k][j] += c * A[i][j];
                }
                b[k] += c * b[i];
            }
        }

        std::vector<double> x(n);
        for (int i = n - 1; i >= 0; --i) {
            x[i] = b[i] / A[i][i];
            for (int k = i - 1; k >= 0; --k) {
                b[k] -= A[k][i] * x[i];
            }
        }
        return x;
    }

    static double mean(const std::vector<double>& data) {
        if (data.empty()) return 0.0;
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }

    static double median(std::vector<double> data) {
        if (data.empty()) return 0.0;
        std::sort(data.begin(), data.end());
        size_t n = data.size();
        if (n % 2 == 0) return (data[n / 2 - 1] + data[n / 2]) / 2.0;
        return data[n / 2];
    }
};

// ==================== 回归核心 ====================

class RegressionCore {
public:
    // 2D 线性回归: y = c0 + c1*x
    static FitResult fitLine2D(const std::vector<Point2D>& pts, const std::vector<int>& indices) {
        FitResult res;
        res.model_degree = 1;

        int n = indices.size();
        if (n < 2) return res;

        double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
        for (int idx : indices) {
            double x = pts[idx].x, y = pts[idx].y;
            sum_x += x; sum_y += y;
            sum_xx += x * x; sum_xy += x * y;
        }

        double denom = n * sum_xx - sum_x * sum_x;
        if (std::abs(denom) < 1e-12) {
            // x 近似常数，退化为水平线
            double y_mean = sum_y / n;
            res.coeffs = { y_mean, 0.0 };
            res.inlier_indices = indices;
            res.rmse = computeRMSE2D(pts, indices, res.coeffs, 1);
            res.r_squared = 0.0;
            return res;
        }

        double a = (n * sum_xy - sum_x * sum_y) / denom;  // c1
        double b = (sum_y * sum_xx - sum_x * sum_xy) / denom; // c0
        res.coeffs = { b, a }; // y = c0 + c1*x
        res.inlier_indices = indices;
        res.rmse = computeRMSE2D(pts, indices, res.coeffs, 1);
        res.r_squared = computeRSquared2D(pts, indices, res.coeffs, 1);
        return res;
    }

    // 2D 多项式回归: y = c0 + c1*x + c2*x^2 + ...
    static FitResult fitPolynomial2D(const std::vector<Point2D>& pts, const std::vector<int>& indices, int degree) {
        FitResult res;
        res.model_degree = degree;

        int n = indices.size();
        if (n < degree + 1) return res;

        int m = degree + 1;
        std::vector<std::vector<double>> ATA(m, std::vector<double>(m, 0.0));
        std::vector<double> ATy(m, 0.0);

        for (int idx : indices) {
            double x = pts[idx].x, y = pts[idx].y;
            std::vector<double> row(m);
            row[0] = 1.0;
            for (int j = 1; j < m; ++j) row[j] = row[j - 1] * x;

            for (int i = 0; i < m; ++i) {
                ATy[i] += row[i] * y;
                for (int j = 0; j < m; ++j) ATA[i][j] += row[i] * row[j];
            }
        }

        res.coeffs = MathUtils::solveLinearSystem(ATA, ATy);
        res.inlier_indices = indices;
        res.rmse = computeRMSE2D(pts, indices, res.coeffs, degree);
        res.r_squared = computeRSquared2D(pts, indices, res.coeffs, degree);
        return res;
    }

    // 3D 平面回归: z = a*x + b*y + c  => coeffs = {a, b, c}
    static FitResult fitPlane3D(const std::vector<Point3D>& pts, const std::vector<int>& indices) {
        FitResult res;
        res.model_degree = 1;

        int n = indices.size();
        if (n < 3) return res;

        double sum_x = 0, sum_y = 0, sum_z = 0;
        double sum_xx = 0, sum_yy = 0, sum_xy = 0, sum_xz = 0, sum_yz = 0;

        for (int idx : indices) {
            double x = pts[idx].x, y = pts[idx].y, z = pts[idx].z;
            sum_x += x; sum_y += y; sum_z += z;
            sum_xx += x * x; sum_yy += y * y; sum_xy += x * y;
            sum_xz += x * z; sum_yz += y * z;
        }

        std::vector<std::vector<double>> A = {
            {sum_xx, sum_xy, sum_x},
            {sum_xy, sum_yy, sum_y},
            {sum_x,  sum_y,  (double)n}
        };
        std::vector<double> b = { sum_xz, sum_yz, sum_z };

        res.coeffs = MathUtils::solveLinearSystem(A, b);
        res.inlier_indices = indices;
        res.rmse = computeRMSE3D(pts, indices, res.coeffs);
        res.r_squared = computeRSquared3D(pts, indices, res.coeffs);
        return res;
    }

    static double predict2D(double x, const std::vector<double>& coeffs, int degree) {
        double y = 0.0, xpow = 1.0;
        for (int i = 0; i <= degree; ++i) {
            y += coeffs[i] * xpow;
            xpow *= x;
        }
        return y;
    }

    static double predict3D(double x, double y, const std::vector<double>& coeffs) {
        return coeffs[0] * x + coeffs[1] * y + coeffs[2];
    }

    static double computeRMSE2D(const std::vector<Point2D>& pts, const std::vector<int>& indices,
        const std::vector<double>& coeffs, int degree) {
        if (indices.empty()) return 0.0;
        double sum = 0.0;
        for (int idx : indices) {
            double diff = predict2D(pts[idx].x, coeffs, degree) - pts[idx].y;
            sum += diff * diff;
        }
        return std::sqrt(sum / indices.size());
    }

    static double computeRSquared2D(const std::vector<Point2D>& pts, const std::vector<int>& indices,
        const std::vector<double>& coeffs, int degree) {
        if (indices.size() < 2) return 0.0;
        double y_mean = 0.0;
        for (int idx : indices) y_mean += pts[idx].y;
        y_mean /= indices.size();

        double ss_res = 0.0, ss_tot = 0.0;
        for (int idx : indices) {
            double pred = predict2D(pts[idx].x, coeffs, degree);
            ss_res += (pts[idx].y - pred) * (pts[idx].y - pred);
            ss_tot += (pts[idx].y - y_mean) * (pts[idx].y - y_mean);
        }
        return (ss_tot < 1e-12) ? 1.0 : 1.0 - ss_res / ss_tot;
    }

    static double computeRMSE3D(const std::vector<Point3D>& pts, const std::vector<int>& indices,
        const std::vector<double>& coeffs) {
        if (indices.empty()) return 0.0;
        double sum = 0.0;
        for (int idx : indices) {
            double diff = predict3D(pts[idx].x, pts[idx].y, coeffs) - pts[idx].z;
            sum += diff * diff;
        }
        return std::sqrt(sum / indices.size());
    }

    static double computeRSquared3D(const std::vector<Point3D>& pts, const std::vector<int>& indices,
        const std::vector<double>& coeffs) {
        if (indices.size() < 2) return 0.0;
        double z_mean = 0.0;
        for (int idx : indices) z_mean += pts[idx].z;
        z_mean /= indices.size();

        double ss_res = 0.0, ss_tot = 0.0;
        for (int idx : indices) {
            double pred = predict3D(pts[idx].x, pts[idx].y, coeffs);
            ss_res += (pts[idx].z - pred) * (pts[idx].z - pred);
            ss_tot += (pts[idx].z - z_mean) * (pts[idx].z - z_mean);
        }
        return (ss_tot < 1e-12) ? 1.0 : 1.0 - ss_res / ss_tot;
    }
};

// ==================== RANSAC 异常点检测 ====================

class RANSACDetector {
public:
    static FitResult detect2D(const std::vector<Point2D>& pts, int degree, double threshold, int maxIter) {
        int n = pts.size();
        if (n < degree + 2) {
            std::vector<int> all(n);
            std::iota(all.begin(), all.end(), 0);
            return (degree == 1) ? RegressionCore::fitLine2D(pts, all)
                : RegressionCore::fitPolynomial2D(pts, all, degree);
        }

        std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        FitResult bestResult;
        int bestInlierCount = -1;
        double bestScore = std::numeric_limits<double>::max();

        int iterations = maxIter;
        int k = 0;

        while (k < iterations) {
            std::vector<int> pool(n);
            std::iota(pool.begin(), pool.end(), 0);
            std::shuffle(pool.begin(), pool.end(), rng);

            std::vector<int> sampleIdx;
            for (int i = 0; i < degree + 1; ++i) sampleIdx.push_back(pool[i]);

            FitResult model = (degree == 1) ?
                RegressionCore::fitLine2D(pts, sampleIdx) :
                RegressionCore::fitPolynomial2D(pts, sampleIdx, degree);

            if (model.coeffs.empty()) continue;

            std::vector<int> inliers;
            for (int i = 0; i < n; ++i) {
                double pred = RegressionCore::predict2D(pts[i].x, model.coeffs, degree);
                if (std::abs(pred - pts[i].y) < threshold) inliers.push_back(i);
            }

            int inlierCount = inliers.size();
            if (inlierCount > bestInlierCount || (inlierCount == bestInlierCount && model.rmse < bestScore)) {
                bestInlierCount = inlierCount;
                bestScore = model.rmse;
                bestResult = model;
                bestResult.inlier_indices = inliers;

                double ratio = (double)inlierCount / n;
                if (ratio > 0.99) {
                    iterations = k + 1;
                }
                else if (ratio > 0) {
                    double p = 0.99;
                    double newIter = std::log(1 - p) / std::log(1 - std::pow(ratio, degree + 1));
                    if (newIter < iterations && newIter > k) iterations = (int)newIter + 1;
                }
            }
            ++k;
        }

        if (bestInlierCount < 0) {
            std::vector<int> all(n);
            std::iota(all.begin(), all.end(), 0);
            return (degree == 1) ? RegressionCore::fitLine2D(pts, all)
                : RegressionCore::fitPolynomial2D(pts, all, degree);
        }

        // 用全部内点重新拟合精化
        FitResult finalResult = (degree == 1) ?
            RegressionCore::fitLine2D(pts, bestResult.inlier_indices) :
            RegressionCore::fitPolynomial2D(pts, bestResult.inlier_indices, degree);

        std::vector<bool> isInlier(n, false);
        for (int idx : finalResult.inlier_indices) isInlier[idx] = true;
        for (int i = 0; i < n; ++i) if (!isInlier[i]) finalResult.outlier_indices.push_back(i);

        return finalResult;
    }

    static FitResult detect3D(const std::vector<Point3D>& pts, double threshold, int maxIter) {
        int n = pts.size();
        if (n < 4) {
            std::vector<int> all(n);
            std::iota(all.begin(), all.end(), 0);
            return RegressionCore::fitPlane3D(pts, all);
        }

        std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        FitResult bestResult;
        int bestInlierCount = -1;

        int iterations = maxIter;
        int k = 0;

        while (k < iterations) {
            std::vector<int> pool(n);
            std::iota(pool.begin(), pool.end(), 0);
            std::shuffle(pool.begin(), pool.end(), rng);

            std::vector<int> sampleIdx = { pool[0], pool[1], pool[2] };
            FitResult model = RegressionCore::fitPlane3D(pts, sampleIdx);
            if (model.coeffs.empty()) continue;

            std::vector<int> inliers;
            for (int i = 0; i < n; ++i) {
                double pred = RegressionCore::predict3D(pts[i].x, pts[i].y, model.coeffs);
                if (std::abs(pred - pts[i].z) < threshold) inliers.push_back(i);
            }

            if ((int)inliers.size() > bestInlierCount) {
                bestInlierCount = inliers.size();
                bestResult = model;
                bestResult.inlier_indices = inliers;

                double ratio = (double)inliers.size() / n;
                if (ratio > 0.99) iterations = k + 1;
                else if (ratio > 0) {
                    double p = 0.99;
                    double newIter = std::log(1 - p) / std::log(1 - std::pow(ratio, 3));
                    if (newIter < iterations && newIter > k) iterations = (int)newIter + 1;
                }
            }
            ++k;
        }

        if (bestInlierCount < 0) {
            std::vector<int> all(n);
            std::iota(all.begin(), all.end(), 0);
            return RegressionCore::fitPlane3D(pts, all);
        }

        FitResult finalResult = RegressionCore::fitPlane3D(pts, bestResult.inlier_indices);
        std::vector<bool> isInlier(n, false);
        for (int idx : finalResult.inlier_indices) isInlier[idx] = true;
        for (int i = 0; i < n; ++i) if (!isInlier[i]) finalResult.outlier_indices.push_back(i);

        return finalResult;
    }
};

// ==================== 数据 IO ====================

class DataIO {
public:
    static bool loadCSV2D(const std::string& fn, std::vector<Point2D>& pts) {
        std::ifstream f(fn);
        if (!f.is_open()) return false;
        std::string line;
        int idx = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::string xs, ys;
            if (std::getline(ss, xs, ',') && std::getline(ss, ys, ',')) {
                Point2D p{ std::stod(xs), std::stod(ys), idx++ };
                pts.push_back(p);
            }
        }
        return true;
    }

    static bool loadCSV3D(const std::string& fn, std::vector<Point3D>& pts) {
        std::ifstream f(fn);
        if (!f.is_open()) return false;
        std::string line;
        int idx = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::string xs, ys, zs;
            if (std::getline(ss, xs, ',') && std::getline(ss, ys, ',') && std::getline(ss, zs, ',')) {
                Point3D p{ std::stod(xs), std::stod(ys), std::stod(zs), idx++ };
                pts.push_back(p);
            }
        }
        return true;
    }

    static void saveResults2D(const std::string& fn, const FitResult& r, const std::vector<Point2D>& pts) {
        std::ofstream f(fn);
        f << std::fixed << std::setprecision(6);
        f << "# Robust Regression Results (2D)\n";
        f << "# Model Degree: " << r.model_degree << "\n";
        f << "# Equation: y = " << r.coeffs[0];
        if (r.coeffs.size() > 1) f << " + " << r.coeffs[1] << "*x";
        for (size_t i = 2; i < r.coeffs.size(); ++i) f << " + " << r.coeffs[i] << "*x^" << i;
        f << "\n# RMSE: " << r.rmse << "\n# R_squared: " << r.r_squared << "\n";
        f << "# Inliers: " << r.inlier_indices.size() << "\n# Outliers: " << r.outlier_indices.size() << "\n\n";
        f << "# Cleaned Data (Inliers):\nx,y\n";
        for (int idx : r.inlier_indices) f << pts[idx].x << "," << pts[idx].y << "\n";
        f << "\n# Outliers:\nx,y\n";
        for (int idx : r.outlier_indices) f << pts[idx].x << "," << pts[idx].y << "\n";
    }

    static void saveResults3D(const std::string& fn, const FitResult& r, const std::vector<Point3D>& pts) {
        std::ofstream f(fn);
        f << std::fixed << std::setprecision(6);
        f << "# Robust Regression Results (3D)\n";
        f << "# Equation: z = " << r.coeffs[0] << "*x + " << r.coeffs[1] << "*y + " << r.coeffs[2] << "\n";
        f << "# RMSE: " << r.rmse << "\n# R_squared: " << r.r_squared << "\n";
        f << "# Inliers: " << r.inlier_indices.size() << "\n# Outliers: " << r.outlier_indices.size() << "\n\n";
        f << "# Cleaned Data (Inliers):\nx,y,z\n";
        for (int idx : r.inlier_indices) f << pts[idx].x << "," << pts[idx].y << "," << pts[idx].z << "\n";
        f << "\n# Outliers:\nx,y,z\n";
        for (int idx : r.outlier_indices) f << pts[idx].x << "," << pts[idx].y << "," << pts[idx].z << "\n";
    }
};

// ==================== 主程序 ====================

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <input.csv> <dim> [degree] [threshold] [max_iter] [output.csv]\n"
        << "  dim      : 2 or 3\n"
        << "  degree   : polynomial degree for 2D (default: 1, use 0 for auto 1~3)\n"
        << "  threshold: RANSAC residual threshold (-1 for auto, default: -1)\n"
        << "  max_iter : max iterations (default: 1000)\n"
        << "  output   : output file (default: result.csv)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) { printUsage(argv[0]); return 1; }

    std::string inputFile = argv[1];
    int dim = std::stoi(argv[2]);
    int degree = (argc > 3) ? std::stoi(argv[3]) : 1;
    double threshold = (argc > 4) ? std::stod(argv[4]) : -1.0;
    int maxIter = (argc > 5) ? std::stoi(argv[5]) : 1000;
    std::string outputFile = (argc > 6) ? argv[6] : "result.csv";

    if (dim == 2) {
        std::vector<Point2D> points;
        if (!DataIO::loadCSV2D(inputFile, points)) {
            std::cerr << "Error: Cannot load " << inputFile << "\n"; return 1;
        }
        if (points.empty()) { std::cerr << "Error: No data.\n"; return 1; }

        // 自动阈值：基于 y 范围的 5%
        if (threshold < 0) {
            double ymin = points[0].y, ymax = points[0].y;
            for (auto& p : points) { ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y); }
            threshold = (ymax - ymin) * 0.05;
            if (threshold < 1e-6) threshold = 0.1;
        }

        FitResult result;
        if (degree == 0) {
            // 自动模型选择：尝试 1~3 阶，用 BIC 选最优
            double bestBIC = std::numeric_limits<double>::max();
            for (int d = 1; d <= 3; ++d) {
                FitResult r = RANSACDetector::detect2D(points, d, threshold, maxIter);
                if (r.coeffs.empty()) continue;
                int k = d + 1;
                int n = r.inlier_indices.size();
                double rss = r.rmse * r.rmse * n;
                double bic = (n > 0) ? n * std::log(rss / n + 1e-12) + k * std::log(n)
                    : std::numeric_limits<double>::max();
                if (bic < bestBIC) { bestBIC = bic; result = r; }
            }
            if (result.coeffs.empty()) result = RANSACDetector::detect2D(points, 1, threshold, maxIter);
        }
        else {
            result = RANSACDetector::detect2D(points, degree, threshold, maxIter);
        }

        if (result.coeffs.empty()) {
            std::cerr << "Error: Failed to fit model.\n"; return 1;
        }

        std::cout << "\n=== 2D Robust Regression ===\n";
        std::cout << "Equation: y = " << result.coeffs[0];
        if (result.coeffs.size() > 1) std::cout << " + " << result.coeffs[1] << "*x";
        for (size_t i = 2; i < result.coeffs.size(); ++i) std::cout << " + " << result.coeffs[i] << "*x^" << i;
        std::cout << "\nRMSE: " << result.rmse << "\nR²:   " << result.r_squared << "\n";
        std::cout << "Inliers:  " << result.inlier_indices.size() << "\n";
        std::cout << "Outliers: " << result.outlier_indices.size() << " (indices: ";
        for (size_t i = 0; i < result.outlier_indices.size(); ++i) {
            if (i) std::cout << ", "; std::cout << result.outlier_indices[i];
        }
        std::cout << ")\n";

        DataIO::saveResults2D(outputFile, result, points);
        std::cout << "Saved to " << outputFile << "\n";

    }
    else if (dim == 3) {
        std::vector<Point3D> points;
        if (!DataIO::loadCSV3D(inputFile, points)) {
            std::cerr << "Error: Cannot load " << inputFile << "\n"; return 1;
        }
        if (points.empty()) { std::cerr << "Error: No data.\n"; return 1; }

        if (threshold < 0) {
            double zmin = points[0].z, zmax = points[0].z;
            for (auto& p : points) { zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z); }
            threshold = (zmax - zmin) * 0.05;
            if (threshold < 1e-6) threshold = 0.1;
        }

        FitResult result = RANSACDetector::detect3D(points, threshold, maxIter);
        if (result.coeffs.empty()) { std::cerr << "Error: Failed to fit model.\n"; return 1; }

        std::cout << "\n=== 3D Robust Regression ===\n";
        std::cout << "Equation: z = " << result.coeffs[0] << "*x + "
            << result.coeffs[1] << "*y + " << result.coeffs[2] << "\n";
        std::cout << "RMSE: " << result.rmse << "\nR²:   " << result.r_squared << "\n";
        std::cout << "Inliers:  " << result.inlier_indices.size() << "\n";
        std::cout << "Outliers: " << result.outlier_indices.size() << " (indices: ";
        for (size_t i = 0; i < result.outlier_indices.size(); ++i) {
            if (i) std::cout << ", "; std::cout << result.outlier_indices[i];
        }
        std::cout << ")\n";

        DataIO::saveResults3D(outputFile, result, points);
        std::cout << "Saved to " << outputFile << "\n";

    }
    else {
        std::cerr << "Error: dim must be 2 or 3.\n"; return 1;
    }

    return 0;
}