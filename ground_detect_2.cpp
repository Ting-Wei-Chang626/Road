#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>
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
        vector<Point2f> inlier;
        vector<Point2f> outlier;
        vector<Point2f> inlier_2;
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

    cv::Size s = a.size();
    double ground_right_up[] = {s.width*0.5,s.height*0.8};
    double ground_left_bottom[] = {s.width*1.0,s.height*0.2};

    vector<KeyPoint> kp3;
    vector<float> des3_temp;

    // To check whether feature point in the ground range 
    for (unsigned index=0; index!=kp1.size(); index++){
        Point2f coord = kp1[index].pt;
        if ((coord.y>ground_right_up[0]) && (coord.y<ground_left_bottom[0]) &&
            (coord.x<ground_right_up[1]) && (coord.x>ground_left_bottom[1])){
                kp3.push_back(kp1[index]);
                des3_temp.push_back(des1.data[index]);
            }
    }

    
    Mat des3;
    des3 = convertVector2Mat(des3_temp, kp3.size(), 1);

    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( des3, des1, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < des3.rows; i++ ){
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    // printf("-- Max dist : %f \n", max_dist );
    // printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    for( int i = 0; i < des3.rows; i++ ){
        if( matches[i].distance < 3*min_dist ){
            good_matches.push_back( matches[i]); }
    }

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( int i = 0; i < good_matches.size(); i++ ){
    //-- Get the keypoints from the good matches
        obj.push_back( kp3[ good_matches[i].queryIdx ].pt );
        scene.push_back( kp2[ good_matches[i].trainIdx ].pt );
  }

    Mat h = findHomography( obj, scene, RANSAC );
    
    return h;
}

void Image::set_outlier_inlier(Mat img1, Mat img2, Mat h, double radius){

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

    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( des1, des2, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < des1.rows; i++ ){
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    // printf("-- Max dist : %f \n", max_dist );
    // printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    for( int i = 0; i < des1.rows; i++ ){
        if( matches[i].distance < 3*min_dist ){
            good_matches.push_back( matches[i]); }
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

    for (unsigned index=0; index!=obj.size(); index++){
        double dis = pow((temp[index].y-scene[index].y),2)+pow((temp[index].x-scene[index].x),2);
    
        if (dis > radius*radius){
            outlier.push_back(obj[index]);
            outlier_2.push_back(scene[index]);
        }else{
            inlier.push_back(obj[index]);
            inlier_2.push_back(scene[index]);
        }
    }
    H = compute_Homography(inlier, inlier_2);
}





class ImageSaver{
    private:
        vector<Image*> images;
    public:
        void add_image(Image* img){
            images.push_back(img);
        }
        ImageSaver(){}
        ~ImageSaver(){}
};

 



class Ground{
    private:
        vector<int[]> ground_pts;
};






int main(){
    ImageSaver saver;
    cout << "successful!!!" << endl;
    // Image* img_initial;
    // Image* img1;
    // saver.add_image(img_initial);
    // saver.add_image(img1);

    // int n = 0;
    // while(0){
        
    // }

    // // if inlier < 4, homography(n) = homography(n-1)
    // while (img_initial->get_inlier().size() < 4){

    // }

    return 0;
}



