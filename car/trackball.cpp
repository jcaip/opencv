#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <numeric>
#include <string>
#include <chrono>

//Track the red ball and try to compute the distance at which it will land. 

//Plan
//Take in input from webcam
//Apply HSV filter to only get red objects
//Contour to detect spheres
//Plot path
//Do physics

using namespace cv;
using namespace std;
double sizeBall = 4; //cm

int iLowH  =  0;   
int iHighH =  180;   
int iLowS  =  150;   
int iHighS = 255;   
int iLowV  =  170;   
int iHighV = 255;   

int printError(String s)
{
    cout << "ERROR: "+s << endl;
    return -1;
}
void printVec(vector<double>& a)
{
    for (auto i : a)
        cout <<i <<" ";
    cout<<endl;
}

void morphFiltering(Mat& input, int a, int b)
{
        auto elementopen = getStructuringElement(MORPH_ELLIPSE, Size(a,a));
        auto elementclose = getStructuringElement(MORPH_ELLIPSE, Size(b,b));
        //morphological opening (remove small objects from the foreground)
        erode(input, input, elementopen);
        dilate(input, input, elementopen);

        //morphological closing (fill small holes in the foreground)
        dilate(input, input, elementclose);
        erode(input, input, elementclose);
}

int main(int argc, char** argv)
{
    //Open video feed
    VideoCapture cap(0);
    if (!cap.isOpened())
       return printError("Cannot open camera");

    //HSV Filter
    //control window 
    Mat imgStart;
    Mat imgStartThresh;
    cap.read(imgStart);
    cap.read(imgStart);
    cap.read(imgStartThresh);
    cvtColor(imgStartThresh, imgStartThresh, COLOR_BGR2HSV); //convert captured frame into HSV
    inRange(imgStartThresh, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgStartThresh); //Threshold the image with the upper range
    morphFiltering(imgStartThresh, 5, 25);
    
    
    for(int i =0 ;i <30;i++)
    {
        cap.read(imgStart);
        cvtColor(imgStart, imgStart, COLOR_BGR2HSV); //convert captured frame into HSV
        inRange(imgStart, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgStart); //Threshold the image with the upper range
        morphFiltering(imgStart, 5, 25);
        bitwise_or(imgStartThresh,imgStart, imgStartThresh);
    }

    SimpleBlobDetector::Params params; //generate blob detector
    params.filterByColor = true;
    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByArea = false;
    params.filterByConvexity = false;
    params.blobColor = 255;
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    //create masks to correctly identify potential mixups
    vector<KeyPoint> all_keypoints;
    vector<KeyPoint> keypoints;
    imshow("start bad stff", imgStartThresh);

    //important frames
    vector<Mat> frames;
    vector<Mat> framesT;
    vector<chrono::high_resolution_clock::time_point> times;
    cap.read(imgStart);

    do 
    {
        Mat imgOriginal;
 
        bool readSuccess = cap.read(imgOriginal);
        if (!readSuccess)
            return printError("Could not read frame from camera");
        Mat imgDiff(imgOriginal);        
        //absdiff(imgOriginal, imgStart, imgDiff); //subtract frame to detect motion

        Mat imgHSV;
        cvtColor(imgDiff, imgHSV, COLOR_BGR2HSV); //convert captured frame into HSV
        Mat imgThreshold; 
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThreshold); //Threshold the image with the upper range
        morphFiltering(imgThreshold,5, 30); 
        Mat subtractor;
        bitwise_and(imgStartThresh, imgThreshold, subtractor);
        absdiff(imgThreshold, subtractor, imgThreshold);

        //update prev
        if(false)
        {
            Mat imgUpdate;
            bitwise_not(imgThreshold,imgUpdate);
            imshow("imgupdate", imgUpdate);
            cvtColor(imgUpdate, imgUpdate, CV_GRAY2BGR);
            bitwise_and(imgOriginal, imgUpdate, imgUpdate);
            imgStart = imgUpdate;
        }

        //blob detector
        detector->detect(imgThreshold, keypoints);
        if(!keypoints.empty())
        {
            frames.push_back(imgDiff);
            framesT.push_back(imgThreshold);
            times.push_back(chrono::high_resolution_clock::now()); 
            all_keypoints.insert(all_keypoints.end(), keypoints.begin(), keypoints.end());
        }
        drawKeypoints(imgOriginal, keypoints, imgOriginal, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        imshow("Morphed Image", imgThreshold); //show the thresholded image
        imshow("Original", imgOriginal); //show the original image
        imshow("Diff", imgDiff); //show the original image

        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout<<"Recording stopped"<<endl;
            break;
        }
    }
    while(true);
    
    vector<double> ballF;
    vector<double> timesF; 
    vector<double> veloF; 
    vector<double> accelF; 
    vector<double> posF; 

    double ballsize = 0.0; 
    double ans = 0.0;
    
    //compute ball size and positions
    for( auto i : all_keypoints)
    {
        posF.push_back(i.pt.y);
        ballsize += i.size;
        ballF.push_back(i.size);
    }
    ballsize = ballsize/all_keypoints.size();
    ballsize /= sizeBall;


    cout<<ballsize<<endl;

    for(size_t i=0;i < posF.size();i++)
        posF[i] /= 100*ballsize;

    //compute deltaT
    for(size_t i = 1; i < times.size(); i++ )
        timesF.push_back((chrono::duration_cast<chrono::milliseconds>(times[i]-times[i-1]).count()));
    for(size_t i=0;i < timesF.size();i++)
        timesF[i] /= 1000.0;
    //velocity
    for(size_t i = 1; i <posF.size();i++)
        veloF.push_back(ballsize*(posF[i]-posF[i-1])/timesF[i-1]);

    //acceleration
    for(size_t i = 1; i <veloF.size();i++)
        accelF.push_back((veloF[i]-veloF[i-1]));

    printVec(ballF);
    printVec(timesF);
    printVec(posF);
    printVec(veloF);
    printVec(accelF);
    
    for(auto i : accelF)
        ans+=i;
    ans/=accelF.size();
    cout<<ans<<" m/s"<<endl;
    destroyAllWindows();

    for(size_t i = 0; i<frames.size();)
    {
        imshow("IMAGE", frames[i]);
        imshow("THRESH", framesT[i]);
        if(waitKey(0) == 110)
            i++;
    }
    return 0;
}



