//
// Created by xwd on 2022/11/18.
//

#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
#include <unistd.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <set>
#include <string>

#include "Sort.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define CNUM 200

void TestSORT(string seqName, bool display);

int main()
{
    bool display = true;
    vector<string> sequences = { "PETS09-S2L1", "TUD-Campus", "TUD-Stadtmitte", "ETH-Bahnhof", "ETH-Sunnyday", "ETH-Pedcross2", "KITTI-13", "KITTI-17", "ADL-Rundle-6", "ADL-Rundle-8", "Venice-2" };
    for (auto seq : sequences)
    {
        TestSORT(seq, display);
        cout << "Process: " << seq << "  Done." << endl;
    }
//    TestSORT("PETS09-S2L1", true);
    return 0;
}

void TestSORT(string seqName, bool display)
{
    cout << "Processing " << seqName << "..." << endl;

    // 0. randomly generate colors, only for display
    RNG rng(0xFFFFFFFF);
    Scalar_<int> randColor[CNUM];
    for (int i = 0; i < CNUM; i++)
        rng.fill(randColor[i], RNG::UNIFORM, 0, 256);

    string imgPath = "/media/calyx/Windy/wdy/data_online/MOT15/train/" + seqName + "/img1/";

    if (display)
        if (access(imgPath.c_str(), 0) == -1)
        {
            cerr << "Image path not found!" << endl;
            display = false;
        }

    // 1. read detection file
    ifstream detectionFile;
    string detFileName = "../data/" + seqName + "/det.txt";
    detectionFile.open(detFileName);

    if (!detectionFile.is_open())
    {
        cerr << "Error: can not find file " << detFileName << endl;
        return;
    }

    string detLine;
    istringstream ss;
    vector<TrackingBox> detData;
    char ch;
    float tpx, tpy, tpw, tph;

    while ( getline(detectionFile, detLine) )
    {
        TrackingBox tb;

        ss.str(detLine);
        ss >> tb.frame >> ch >> tb.id >> ch;
        ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph;
        ss.str("");

        tb.box = Rect_<float>(Point_<float>(tpx, tpy), Point_<float>(tpx + tpw, tpy + tph));
        detData.push_back(tb);
    }
    detectionFile.close();

    // 2. group detData by frame
    int maxFrame = 0;
    for (auto tb : detData) //find max frame number
    {
        if (maxFrame < tb.frame)
            maxFrame = tb.frame;
    }

    ////////////////////
    // 设置SORT算法初始化参数
    int max_age = 5;
    int min_hits = 2;
    double iouThreshold = 0.1;

    Sort MOT_tracker(max_age, min_hits, iouThreshold);

    //初始化返回的vector
    vector<TrackingBox> frameTrackingResult;

    ///////////////////
//    vector<vector<TrackingBox>> detFrameData;
    vector<TrackingBox> tempVec;
    for (int fi = 0; fi < maxFrame; fi++)
    {
        for (auto tb : detData)
            if (tb.frame == fi + 1) // frame num starts from 1
                tempVec.push_back(tb);

        frameTrackingResult = MOT_tracker.update(tempVec);

        // 得到结果
        if (display) // read image, draw results and show them
        {
            ostringstream oss;
            oss << imgPath << setw(6) << setfill('0') << fi + 1;
            Mat img = imread(oss.str() + ".jpg");
            if (img.empty())
                continue;

            for (auto tb : frameTrackingResult) {
                cv::rectangle(img, tb.box, randColor[tb.id % CNUM], 2, 8, 0);
                cv::putText(img, "ID:" + std::to_string(tb.id), cv::Point(int(tb.box.x), int(tb.box.y) - 10), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            randColor[tb.id % CNUM], 2);
            };
            cv::putText(img, "Frame: " + std::to_string(fi + 1), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 0, 255), 2);
            imshow(seqName, img);
            waitKey(0);

        }
        tempVec.clear();
    }

    if (display)
        destroyAllWindows();
}

