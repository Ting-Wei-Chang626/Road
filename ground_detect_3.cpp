#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <queue>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

class Image;
class ImageSaver;

template<typename _Tp>
vector<_Tp> convertMat2Vector(const Mat &mat)
{
	return (vector<_Tp>)(mat.reshape(1, 1));//通道數不變，按行轉爲一行
}
 
/****************** vector轉Mat *********************/
template<typename _Tp>
cv::Mat convertVector2Mat(vector<_Tp> v, int channels, int rows)
{
    
	cv::Mat mat = cv::Mat(v).clone();//將vector變成單列的mat
	cv::Mat dest = mat.reshape(channels, rows);
	return dest;
}


class Image{
    private:
        Mat img;
        Mat img_draw;
        Mat H; // Homography compute by image(n) and image(n-1) 
        vector<Point2f> inlier;  // points of frame(n-1)
        vector<Point2f> outlier;
        vector<Point2f> inlier_2; // points of frame(n)
        vector<Point2f> outlier_2;
    public:
        Image(Mat i):img(i){}
        ~Image(){}
        void set_homography(Mat h){
            H = h;
        }
        void set_img(Mat image){
            img = image;
        }
        Mat get_image(){
            return img;
        }
        Mat get_draw_image(){
            return img_draw;
        }
        Mat get_homography(){
            return H;
        }
        Mat compute_first_Homography(Mat a, Mat b);
        Mat compute_Homography(vector<Point2f> a, vector<Point2f> b);

        void set_outlier_inlier(Mat img1, Mat img2, Mat H, double radius);
        

        vector<Point2f> get_inlier(){
            return inlier;
        }
        vector<Point2f> get_outlier(){
            return outlier;
        }
        
        vector<Point2f> get_inlier_2(){
            return inlier_2;
        }
        vector<Point2f> get_outlier_2(){
            return outlier_2;
        }

        void draw_inlier(vector<Point2f> toDraw){
            for (auto pts=toDraw.begin(); pts!=toDraw.end(); pts++){
                img_draw = img;
                cv::circle(img_draw, cvPoint((int)(pts->x), (int)(pts->y)), 3, (0,255,0), 3);
            }
        }

        void draw_outlier(vector<Point2f> toDraw){
            for (auto pts=toDraw.begin(); pts!=toDraw.end(); pts++){
                img_draw = img;
                cv::circle(img_draw, cvPoint((int)(pts->x), (int)(pts->y)), 3, (0,255,0), 3);
            }
        }
};


Mat Image::compute_Homography(vector<Point2f> a, vector<Point2f> b){
    // cout << "findHomography" << endl;
    Mat h = findHomography( a, b, RANSAC);
    return h;
}

Mat Image::compute_first_Homography(Mat a, Mat b){
    // initialize surf
    int minHessian = 400;
    
    Ptr<xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
    std::vector<KeyPoint> kp1;
    std::vector<KeyPoint> kp2;
    Mat des1;//descriptor  
    Mat des2;//descriptor  
    
    // using surf to compute feature point and its description
    surf->detectAndCompute(a, Mat(), kp1, des1);
    surf->detectAndCompute(b, Mat(), kp2, des2);

    // if(des1.type()!=CV_32F) { 
    //     des1.convertTo(des1, CV_32F); 
    // } 
    // if(des2.type()!=CV_32F) { 
    //     des2.convertTo(des2, CV_32F);
    // }

    // Define the box range of ground

        /////////////////////
        //                 //
        //                 //
        //  /////////////  //
        //  //         //  //
        //  // Ground  //  //
        //  //         //  //
        /////////////////////

    cv::Size s = a.size();
    double ground_right_up[] = {s.height*0.5, s.width*0.8};
    double ground_left_bottom[] = {s.height*1.0, s.width*0.2};

    // cout << *ground_right_up << endl;
    // cout << *ground_left_bottom << endl;


    vector<KeyPoint> kp3;
    vector<int> des3_temp;

    // To check whether feature point in the ground range 
    for (int index=0; index!=kp1.size(); index++){
        Point2f coord = kp1[index].pt;
        if ((coord.y>ground_right_up[0]) && (coord.y<ground_left_bottom[0]) &&
            (coord.x<ground_right_up[1]) && (coord.x>ground_left_bottom[1])){
                kp3.push_back(kp1[index]);
                des3_temp.push_back(index);
            }
    }

    Mat des3 = Mat::zeros(kp3.size(), des1.cols, CV_32F);
    // cout << des3.size() <<endl;
    // cout << des3.rows << endl;

    for (int index=0; index!=des3_temp.size(); index++){
        des3.row(index)=des1.row(des3_temp[index])+0;
    }

    // cout << "Here" <<endl;
    // cout << kp1.size() << endl;
    // cout << kp3.size() << endl;
    // cout << des3.size() << endl;
    // cout << des1.size() << endl;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch>  > matches;
    // FlannBasedMatcher matcher;
    // std::vector< DMatch > matches;
    matcher->knnMatch( des3, des1, matches, 2);

 
    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    // cout << good_matches.size() << endl;

    for( int i = 0; i < good_matches.size(); i++ ){
    //-- Get the keypoints from the good matches
        obj.push_back( kp3[ good_matches[i].queryIdx ].pt );
        scene.push_back( kp2[ good_matches[i].trainIdx ].pt );
  }
    // cout << obj <<endl;
    // cout << "findHomography" << endl;
    Mat h = findHomography( obj, scene, RANSAC );
    
    return h;
}

void Image::set_outlier_inlier(Mat img1, Mat img2, Mat h, double radius=15){

    /*static Ptr<SURF> cv::xfeatures2d::SURF::create	(	double 	hessianThreshold = 100,
    int 	nOctaves = 4,
    int 	nOctaveLayers = 3,
    bool 	extended = false,
    bool 	upright = false 
    )*/	

    int minHessian = 400; // defult = 100	
    Ptr<xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
    std::vector<KeyPoint> kp1;
    std::vector<KeyPoint> kp2;
    Mat des1;//descriptor  
    Mat des2;//descriptor  
    
    // using surf to compute feature point and its description
    surf->detectAndCompute(img1, Mat(), kp1, des1);
    surf->detectAndCompute(img2, Mat(), kp2, des2);

    // if (des1.type()!=CV_32F) { 
    //     des1.convertTo(des1, CV_32F); 
    // } 

    // if (des2.type()!=CV_32F) { 
    //     des2.convertTo(des2, CV_32F); 
    // } 

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > matches;

    matcher->knnMatch( des1, des2, matches, 2);

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( int i = 0; i < good_matches.size(); i++ ){
    //-- Get the keypoints from the good matches
        obj.push_back( kp1[ good_matches[i].queryIdx ].pt );
        scene.push_back( kp2[ good_matches[i].trainIdx ].pt );
    }

    std::vector<Point2f> temp;
    perspectiveTransform(obj, temp, h);

    std::vector<Point2f> outlier_temp;
    std::vector<Point2f> outlier_2_temp;
    std::vector<Point2f> inlier_temp;
    std::vector<Point2f> inlier_2_temp;

    cout << "dis" <<endl;

    for (unsigned index=0; index!=obj.size(); index++){
        double dis = pow((temp[index].y-scene[index].y),2)+pow((temp[index].x-scene[index].x),2);
    
        if (dis > radius*radius){
            outlier_temp.push_back(obj[index]);
            outlier_2_temp.push_back(scene[index]);
        }else{
            inlier_temp.push_back(obj[index]);
            inlier_2_temp.push_back(scene[index]);
        }
    }

    inlier = inlier_temp;
    inlier_2 = inlier_2_temp;
    outlier = outlier_temp;
    outlier_2 = outlier_2_temp;

    // H = compute_Homography(inlier, inlier_2);
}




// To save the images
class ImageSaver{
    private:
        queue<Image*> images;
    public:
        void add_image(Image* img){
            images.push(img);
        }
        queue<Image*> get_image_queue(){
            return images;
        }

        ImageSaver(){}
        ~ImageSaver(){}
};




int main(){
    ImageSaver saver;
    cout << "successful!!!" << endl;

    // 1st and 2nd image to define homography (user define ground)
    Mat img0 = imread("IMG_20200624_100203.jpg");
    Image frame_0(img0);
    Mat img1 = imread("IMG_20200624_100204.jpg");
    Image frame_1(img1);

    // compute user-defined homography of ground
    Mat H = frame_0.compute_first_Homography(frame_0.get_image(), frame_1.get_image());

    frame_1.set_homography(H);

    // frame_1.set_outlier_inlier(frame_0.get_image(), frame_1.get_image(), frame_1.get_homography(), 100);

    // frame_1.draw_inlier(frame_1.get_inlier_2());

    // namedWindow("Result", CV_WINDOW_NORMAL);
    // imshow("Result", frame_1.get_draw_image());
    // waitKey(0);
    // destroyAllWindows();


    // 3rd and 4th image
    Mat img2 = imread("IMG_20200624_100205.jpg");
    Image frame_2(img2);
    // Mat img3;
    // Image frame_3(img3);


    double radius = 60;
    frame_2.set_outlier_inlier(frame_1.get_image(), frame_2.get_image(), frame_1.get_homography(), radius);

    // At least 8 pts to define homography
    while (frame_2.get_inlier().size() < 8){
        radius += 2;
        frame_2.set_outlier_inlier(frame_1.get_image(), frame_2.get_image(), frame_1.get_homography(), radius);
    }

    Mat H1 = frame_2.compute_Homography(frame_2.get_inlier(), frame_2.get_inlier_2());
    frame_2.set_homography(H1);
    
    
    frame_2.draw_outlier(frame_2.get_outlier_2());
    frame_2.draw_inlier(frame_2.get_inlier_2());

    namedWindow("Result", CV_WINDOW_NORMAL);
    imshow("Result", frame_2.get_draw_image());
    waitKey(0);
    destroyAllWindows();

    return 0;
}


/*

int main(){
    ImageSaver saver;
    cout << "successful!!!" << endl;


    // 1st and 2nd image to define homography (user define ground)
    Mat img0;
    Image frame_0(img0);
    Mat img1;
    Image frame_1(img1);

    // compute user-defined homography of ground
    Mat H = frame_0.compute_first_Homography(frame_0.get_image(), frame_1.get_image());
    frame_1.set_homography(H);


    while(1){
        saver.add_image(&frame_1);
        Mat img2; // call next frame
        Image* frame_2 = new Image(img2);
        
        double radius = 4;
        frame_2->set_outlier_inlier(saver.get_image_queue().back()->get_image(), frame_2->get_image(),
         saver.get_image_queue().back()->get_homography(), radius);

        // At least 8 pts to define homography
        while (frame_2->get_inlier().size() < 8){
            radius++;
            frame_2->set_outlier_inlier(saver.get_image_queue().back()->get_image(), frame_2->get_image(),
             saver.get_image_queue().back()->get_homography(), radius);
        }

        Mat H1 = frame_2->compute_Homography(frame_2->get_inlier(), frame_2->get_inlier_2());
        frame_2->set_homography(H1);
    
    
        frame_2->draw_outlier(frame_2->get_outlier_2());
        frame_2->draw_inlier(frame_2->get_inlier_2());

        imshow("Result", frame_2->get_draw_image());
        waitKey(0);
        destroyAllWindows();

        saver.add_image(frame_2);
        saver.get_image_queue().pop();
    }

    return 0;
}

*/

