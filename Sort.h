//
// Created by xwd on 2022/11/17.
//

#ifndef SORT_CPP_SORT_H
#define SORT_CPP_SORT_H

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "Hungarian.h"
#include "KalmanTracker.h"

using namespace cv;

struct TrackingBox
{
    int frame;
    int id;
    Rect_<float> box;
};

class Sort
{
public:
    Sort(int maxAge, int minHits, double iou_th);
    ~Sort();

    vector<TrackingBox> update(std::vector<TrackingBox> detFrameData);

    int max_age;
    int min_hits;
    double iouThreshold;
    vector<KalmanTracker> trackers;

private:
    double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);
};

#endif //SORT_CPP_SORT_H