//
// Created by xwd on 2022/11/17.
//

#include <set>
#include "Sort.h"

Sort::Sort(int maxAge, int minHits, double iou_th) {
    max_age = maxAge;
    min_hits = minHits;
    iouThreshold = iou_th;

    KalmanTracker::kf_count = 0; // 初始化
}
Sort::~Sort() {}

vector<TrackingBox> Sort::update(vector<TrackingBox> detFrameData) {

    // 输入为当前帧的dets,在这里用CV的Rect结构

    int frame_count = 0;

    // variables used in the for-loop
    vector<Rect_<float>> predictedBoxes;
    vector<vector<double>> iouMatrix;
    vector<int> assignment;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;
    vector<TrackingBox> frameTrackingResult;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;


    if (trackers.size() == 0) // the first frame met
    {
        // initialize kalman trackers using first detections.
        for (unsigned int i = 0; i < detFrameData.size(); i++)
        {
            KalmanTracker trk = KalmanTracker(detFrameData[i].box);
            trackers.push_back(trk);
        }

        // output the first frame detections
        for (unsigned int id = 0; id < detFrameData.size(); id++)
        {
            TrackingBox tb = detFrameData[id];
        }
    }

    else {

        // 1.遍历当前存在的trackers, 去掉predict后无用的tracker, get predicted locations from existing trackers.
        predictedBoxes.clear();

        for (auto it = trackers.begin(); it != trackers.end();) {
            Rect_<float> pBox = (*it).predict();//每个活跃的tracker先进行预测
            if (pBox.x >= 0 && pBox.y >= 0) {
                predictedBoxes.push_back(pBox);
                it++;
            } else {
                it = trackers.erase(it);
            }
        }

        // 2. trackers 与 dets 的关联匹配  associate detections to tracked object (both represented as bounding boxes)
        trkNum = predictedBoxes.size();
        detNum = detFrameData.size();

        iouMatrix.clear();
        iouMatrix.resize(trkNum, vector<double>(detNum, 0));

        for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
        {
            for (unsigned int j = 0; j < detNum; j++) {
                // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[j].box);
            }
        }

        // solve the assignment problem using hungarian algorithm.
        // the resulting assignment is [track(prediction) : detection], with len=preNum
        HungarianAlgorithm HungAlgo;
        assignment.clear();
        HungAlgo.Solve(iouMatrix, assignment);

        // find matches, unmatched_detections and unmatched_predictions
        unmatchedTrajectories.clear();
        unmatchedDetections.clear();
        allItems.clear();
        matchedItems.clear();

        if (detNum > trkNum) //	there are unmatched detections
        {
            for (unsigned int n = 0; n < detNum; n++)
                allItems.insert(n);

            for (unsigned int i = 0; i < trkNum; ++i)
                matchedItems.insert(assignment[i]);

            set_difference(allItems.begin(), allItems.end(),
                           matchedItems.begin(), matchedItems.end(),
                           insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        } else if (detNum < trkNum) // there are unmatched trajectory/predictions
        {
            for (unsigned int i = 0; i < trkNum; ++i)
                if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.insert(i);
        } else;

        // filter out matched with low IOU
        matchedPairs.clear();
        for (unsigned int i = 0; i < trkNum; ++i) {
            if (assignment[i] == -1) // pass over invalid values
                continue;
            if (1 - iouMatrix[i][assignment[i]] < iouThreshold) {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            } else
                matchedPairs.push_back(cv::Point(i, assignment[i]));
        }

        // 3. 更新trackers, updating trackers

        // update matched trackers with assigned detections.
        // each prediction is corresponding to a tracker

        int detIdx, trkIdx;
        for (unsigned int i = 0; i < matchedPairs.size(); i++) {
            trkIdx = matchedPairs[i].x;
            detIdx = matchedPairs[i].y;
            trackers[trkIdx].update(detFrameData[detIdx].box);
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd: unmatchedDetections) {
            KalmanTracker tracker = KalmanTracker(detFrameData[umd].box);
            trackers.push_back(tracker);
        }
    }

    // get trackers' output
    frameTrackingResult.clear();
    for (auto it = trackers.begin(); it != trackers.end();)
    {
//        if (((*it).m_time_since_update < 1) &&
//            ((*it).m_hit_streak >= min_hits || frame_count <= min_hits)) // frame_count因素除开
        if (((*it).m_time_since_update < 1) &&
            ((*it).m_hit_streak >= min_hits))
        {
            TrackingBox res;
            res.box = (*it).get_state();
            res.id = (*it).m_id + 1;
            res.frame = frame_count;
            frameTrackingResult.push_back(res);
            it++;
        }
        else
            it++;

        // remove dead tracklet
        if (it != trackers.end() && (*it).m_time_since_update > max_age)
            it = trackers.erase(it);
    }

    return frameTrackingResult;
}



// Computes IOU between two bounding boxes
double Sort::GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}