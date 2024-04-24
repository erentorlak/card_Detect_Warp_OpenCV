#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
// Function declarations
std::vector<cv::Point> orderPoints(const std::vector<cv::Point>& pts){  
    std::vector<cv::Point> rect(4);

    std::vector<int> sums(pts.size()), diffs(pts.size());
    for (size_t i = 0; i < pts.size(); i++) {
        sums[i] = pts[i].x + pts[i].y;
        diffs[i] = pts[i].x - pts[i].y;
    }

    // Top-left point will have the smallest sum...
    rect[0] = pts[std::min_element(sums.begin(), sums.end()) - sums.begin()];
    // Bottom-right point will have the largest sum...
    rect[2] = pts[std::max_element(sums.begin(), sums.end()) - sums.begin()];
    // Top-right point will have the smallest difference...
    rect[1] = pts[std::min_element(diffs.begin(), diffs.end()) - diffs.begin()];
    // Bottom-left point will have the largest difference...
    rect[3] = pts[std::max_element(diffs.begin(), diffs.end()) - diffs.begin()];

    return rect;
}

cv::Mat fourPointTransform(const cv::Mat& image, std::vector<cv::Point>& pts){
    std::vector<cv::Point2f> srcPts;
    for (const auto& pt : pts) {
        srcPts.push_back(cv::Point2f(pt.x, pt.y));
    }

    // Determine the "width" and "height" of the new image
    float widthA = std::sqrt(std::pow(srcPts[2].x - srcPts[3].x, 2) + std::pow(srcPts[2].y - srcPts[3].y, 2));
    float widthB = std::sqrt(std::pow(srcPts[1].x - srcPts[0].x, 2) + std::pow(srcPts[1].y - srcPts[0].y, 2));
    float width = std::max(widthA, widthB);

    float heightA = std::sqrt(std::pow(srcPts[1].x - srcPts[2].x, 2) + std::pow(srcPts[1].y - srcPts[2].y, 2));
    float heightB = std::sqrt(std::pow(srcPts[0].x - srcPts[3].x, 2) + std::pow(srcPts[0].y - srcPts[3].y, 2));
    float height = std::max(heightA, heightB);

    // Ensure that the longer dimension is always used as the height for vertical alignment
    bool isHorizontal = width > height;
    if (isHorizontal) {
        std::swap(width, height);
    }

    // srcPts should be [top-left, top-right, bottom-right, bottom-left]
    if (isHorizontal) {
        // Assume the card is horizontal, reorder the points for a vertical output
        std::rotate(srcPts.begin(), srcPts.begin() + 1, srcPts.end()); // Rotate left
    }

    std::vector<cv::Point2f> dstPts = {
        cv::Point2f(0, 0),
        cv::Point2f(0, height - 1),  // Start from top-left, go down to bottom-left
        cv::Point2f(width - 1, height - 1),  // Bottom-left to bottom-right
        cv::Point2f(width - 1, 0)  // Bottom-right to top-right
    };

    cv::Mat M = cv::getPerspectiveTransform(srcPts, dstPts);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(width, height));  // Size set with height greater than width

    return warped;
}


int main() {
    cv::VideoCapture cap(0); 

    if (!cap.isOpened()) {
        return -1;
    }

    while (true) {
        cv::Mat frame, gray, binary, morphed, finalContours;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cap >> frame; 
        if (frame.empty()) { break; }

        // Converting to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Applying Gaussian Blur
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);    // 5,5 is the kernel size and 0 is the standard deviation
        cv::imshow("Gray - 1", gray);

        // Applying adaptive thresholding
        cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 5, 2); //5 is the block size, 2 is the Constant subtracted from the mean
        cv::imshow("Binary - 2", binary);
        
        // Finding contours with RETR_EXTERNAL flag
        cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);//RETR_EXTERNAL retrieves only the outermost contours

        std::vector<std::vector<cv::Point>> filteredContours;
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) > 100) {
                filteredContours.push_back(contour);
            }
        }

        // Draw filtered contours
        binary = cv::Mat::zeros(binary.size(), CV_8UC1);    // Create a black image 
        cv::drawContours(binary, filteredContours, -1, cv::Scalar(255), -1);        // Draw the filtered contours
        cv::imshow("Contours - 3", binary); 

        // Morphological operations to clean up noise
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));   // Create a 3x3 kernel
        cv::erode(binary, morphed, kernel, cv::Point(-1, -1), 6);   // Erode the image 6 times
        cv::dilate(morphed, morphed, kernel, cv::Point(-1, -1), 6); // Dilate the image 6 times

        cv::findContours(morphed, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // Find contours again
        cv::imshow("Morphed - 4", morphed); 

        
        std::vector<std::vector<cv::Point>> validPolygons;  
        std::vector<double> polygonAreas;   
        
        for (const auto& contour : contours) {
            double epsilon = 0.05 * cv::arcLength(contour, true);  // arcLength Computes a curve length or a closed contour perimeter 

            std::vector<cv::Point> approxCurve; 
            cv::approxPolyDP(contour, approxCurve, epsilon, true);
            
            if (approxCurve.size() == 4 && cv::isContourConvex(approxCurve)) {  // Check if the contour is convex and has 4 corners
                double area = cv::contourArea(approxCurve);

                if (area > 1000) {  // Filter out small polygons
                    std::vector<double> edges, angles;

                    for (int i = 0; i < 4; i++) {   // Calculate edge lengths and angles
                        cv::Point p = approxCurve[i];
                        cv::Point q = approxCurve[(i + 1) % 4];
                        double length = cv::norm(p - q);
                        edges.push_back(length);
        
                        cv::Point r = approxCurve[(i + 2) % 4];
                        double angle = std::abs(atan2(q.y - p.y, q.x - p.x) - atan2(r.y - q.y, r.x - q.x)) * 180 / CV_PI;
                        if (angle > 180) angle = 360 - angle;
                        angles.push_back(angle);
                    }
        
                    bool validEdges = (std::max(edges[0], edges[2]) / std::min(edges[0], edges[2]) < 2) &&  // Check edge length ratios
                                      (std::max(edges[1], edges[3]) / std::min(edges[1], edges[3]) < 2);
                    bool validAngles = std::all_of(angles.begin(), angles.end(), [](double angle) {     // Check angle values
                        return angle >= 30 && angle <= 150;     
                    });
        
                    if (validEdges && validAngles) {    // Push valid polygons and their areas
                        validPolygons.push_back(approxCurve);
                        polygonAreas.push_back(area);
                    }
                }
            }
        }
    
        // Calculate mean and standard deviation of areas
        double mean = std::accumulate(polygonAreas.begin(), polygonAreas.end(), 0.0) / polygonAreas.size();
        double sq_sum = std::inner_product(polygonAreas.begin(), polygonAreas.end(), polygonAreas.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / polygonAreas.size() - mean * mean);
    
        // Draw polygons within one standard deviation from the mean
        // If you have diffrent size of polygons, you can use this code to draw polygons within one standard deviation from the mean only
        for (size_t i = 0; i < validPolygons.size(); i++) {
            cv::drawContours(frame, std::vector<std::vector<cv::Point>>{validPolygons[i]}, -1, cv::Scalar(0, 255, 0), 2);
            //if (polygonAreas[i] >= (mean - stdev) && polygonAreas[i] <= (mean + stdev)) {
            //    cv::drawContours(frame, std::vector<std::vector<cv::Point>>{validPolygons[i]}, -1, cv::Scalar(0, 255, 0), 2);
            //}
            //else {
            //    cv::drawContours(frame, std::vector<std::vector<cv::Point>>{validPolygons[i]}, -1, cv::Scalar(0, 0, 255), 2);
            //}
        }
        cv::imshow("Frame", frame);

        std::vector<cv::Mat> cards;
        for (size_t i = 0; i < contours.size(); i++) {

            double area = cv::contourArea(contours[i]);

            std::vector<cv::Point> approx;
            double epsilon = 0.02 * cv::arcLength(contours[i], true);
            cv::approxPolyDP(contours[i], approx, epsilon, true);   // Approximate the contour to a polygon
            
            if (approx.size() == 4) { // Ensure the contour has exactly 4 corners   
                std::vector<cv::Point> hull;
                cv::convexHull(approx, hull); //  Find the convex hull of the polygon
                std::vector<cv::Point> orderedHull = orderPoints(hull); // Order the points in the convex hull
                if (orderedHull.size() == 4) { 
                    cv::Mat card = fourPointTransform(frame, orderedHull);  // Perform perspective transformation
                    if (!card.empty()) {
                        cv::resize(card, card, cv::Size(300, 200)); // Standardize card size
                        cards.push_back(card);
                    }
                }
            }
        }

        // Concatenate images horizontally
        if (!cards.empty()) {
            cv::Mat finalDisplay;
            cv::hconcat(cards.data(), cards.size(), finalDisplay);
            cv::imshow("Cards", finalDisplay);
        }

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}