#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

const int DETECTION_WIDTH = 320;

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

    CascadeClassifier faceDetector;
    faceDetector.load("haarcascade_frontalface_alt.xml");

    while(1)
    {
        Mat frame;
        bool bSuccess = cap.read(frame); //read new frame from webcam
        
        if (!bSuccess)
        {
            cout<<"Failed to get a frame from the webcam"<<endl;
            break;
        }
        if (waitKey(30) ==27)
        {
            cout<<"esc pressed by user"<<endl;
            break;
        }

        Mat img;
        Mat smallImg;

        //MAKE GRAYSCALE
        if (frame.channels() ==3)
            cvtColor(frame, img, CV_BGR2GRAY);
        else if (frame.channels() ==4)
            cvtColor(frame, img, CV_BGR2GRAY);
        else 
            img = frame;

        //MAKE SMALL
        float scale = img.cols / (float) DETECTION_WIDTH;
        if (img.cols > DETECTION_WIDTH) {
            //Shrink image and perserve aspect ratio
            int scaledHeight = cvRound(img.rows/scale);
            resize(img, smallImg, Size(DETECTION_WIDTH, scaledHeight));
        }
        else {
            smallImg = img;
        }
        
        //EQUALIZE
        equalizeHist(smallImg, smallImg);

        int flags = 0|CV_HAAR_SCALE_IMAGE; //search for many faces
        Size minFeatureSize(20,20);// smallest face size
        float searchScaleFactor = 1.1; //how many sizes to searchA
        int minNeighbors = 4;

        std::vector<Rect> faces;
        faceDetector.detectMultiScale(smallImg, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize);
        for (Rect r : faces){
            r.x = cvRound(r.x * scale);
            r.y = cvRound(r.y * scale);
            r.width = cvRound(r.width * scale);
            r.height = cvRound(r.height * scale);
            rectangle(frame, r, Scalar(255,0,0)); 
        }
        imshow("Webcam", frame);
    }
    return 0;
}
