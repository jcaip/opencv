#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    VideoCapture cap(0); // open video camera no0

    if (!cap.isOpened()) //no video camera
    {
        cout<<"Cannot open the video camera"<<endl;
        return -1;
    }

    double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    double dHeight= cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    cout <<"Frame Size: "<< dWidth <<"x" << dHeight<<endl;
    namedWindow("MyWebcam", CV_WINDOW_AUTOSIZE);

    while(1)
    {
        Mat frame;
        bool bSuccess = cap.read(frame); //read new frame from webcam

        if (!bSuccess)
        {
            cout<<"Failed to get a frame from the webcam"<<endl;
            break;
        }
        imshow("Webcam", frame);
        if (waitKey(30) ==27)
        {
            cout<<"esc pressed by user"<<endl;
            break;
        }
    }
    return 0;
}
