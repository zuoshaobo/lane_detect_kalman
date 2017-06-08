#include <iostream>  
#include <opencv/cv.h>  
#include <cxcore.h>    
#include <opencv/highgui.h>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <stdio.h>  
//#include "stdafx.h"  
using namespace std;  
using namespace cv;  
  
  
#define INF 99999999 //用于直线斜率的初始化，代表无穷大  
  
  
//使用播放控制条需要的全局变量  
int        g_slider_position = 0;  
CvCapture* g_capture = NULL;  
int        cur_frame = 0;         //用于指示g_capture的当前帧  
  
  
void onTrackbarSlide(int pos);  //回调函数  
  
  
int main(){  
g_capture = cvCreateFileCapture("project_video.mp4");//使用全局变量抓取视频画面  
//CvCapture *capture = cvCreateFileCapture("1.avi");//注意读取不在工程目录下文件时的路径要加双斜杠  
/*cvCreateFileCapture()通过参数设置确定要读入的avi文件，返回一个指向CvCapture结构的指针。 
这个结构包括了所有关于要读入avi文件的信息，其中包含状态信息。调用这个函数之后，返回指针 
所指向的CvCapture结构被初始化到对应的avi文件的开头。*/  
  
  
IplImage *img = cvQueryFrame(g_capture);  
//cvQueryFrame的参数为CvCapture结构的指针。用来将下一帧视频文件载入内存，返回一个对应当前帧的指针。  
  
  
/************************************* 预处理 ***************************************/  
//读取矩阵  
CvMemStorage *memstorageTest=cvCreateMemStorage(0);  
/*用来创建一个内存存储器，来统一管理各种动态对象的内存。函数返回一个新创建的内存存储器指针。 
参数对应内存器中每个内存块的大小，为0时内存块默认大小为64k。*/  
  
  
CvFileStorage *warp_read=cvOpenFileStorage("1.xml",memstorageTest,CV_STORAGE_READ);//矩阵所在的xml文件名  
/*cvOpenFileStorage打开存在或创建新的文件，第一个参数为矩阵，第二个为存储器，第三 
个为flag，有CV_STORAGE_READ（打开文件读数据）和CV_STORAGE_WRITE（打开文件写数据）*/  
  
CvMat *map_matrix = cvCreateMat(3,3,CV_32FC1);  
CvMat *inverse    = cvCreateMat(3,3,CV_32FC1);  
/*函数 cvCreateMat 为新的矩阵分配头和下面的数据，并且返回一个指向新创建的矩阵的指针。 
参数分别为矩阵行数，列数，矩阵类型。CV_32FC1表示32位浮点型单通道矩阵*/  
  
map_matrix = (CvMat*)cvReadByName(warp_read,NULL,"WarpMatrix",NULL);//读取矩阵，注意双括号里面是矩阵的名字  
cvInvert(map_matrix, inverse, CV_SVD);//求出逆矩阵用以重新投射会原视角  
  
//设置原图像的ROI范围  
int x = 0,y = 157;//这个y值就是要修改的，尽量裁掉天空，不然会给处理带来困难  
int width = img->width , height = 256;  
//视频左上角坐标为（0，0）  
  
  
  
//创建显示窗口  
cvNamedWindow("OriginalView",CV_WINDOW_AUTOSIZE);//原视频  
cvNamedWindow("IPMview",CV_WINDOW_AUTOSIZE);//反透视变换后效果  
cvNamedWindow("AfterCanny",CV_WINDOW_AUTOSIZE);//canny边缘检测后效果  
cvNamedWindow("Erode&Dilate",CV_WINDOW_AUTOSIZE);//腐蚀膨胀后效果  
//cvNamedWindow("AfterSmooth",CV_WINDOW_AUTOSIZE);//高斯模糊后效果  
cvNamedWindow("Hough",CV_WINDOW_AUTOSIZE);//Hough直线变换后效果  
cvNamedWindow("Result",CV_WINDOW_AUTOSIZE);//最终结果  
  
CvMemStorage *storage = cvCreateMemStorage();//内存块，存储中间变量  
CvSeq *lines = 0;//存储Hough变换所得结果  
  
//初始化用于视频播放控制的滑动条  
int frames = (int)cvGetCaptureProperty(  
g_capture,  
CV_CAP_PROP_FRAME_COUNT//以帧数来设置读入位置  
);  
  
  
if(frames != 0){  
cvCreateTrackbar(  
"Frames", //进度条名称  
"Result", //让进度条显示在最终结果的窗口  
&g_slider_position,  
frames,  
onTrackbarSlide//调用一次onTrackbarSlide  
);  
}  
/************************************* 预处理结束 ***************************************/  
  
  
while(1)  
{  
if(!img) break;//视频为空则退出  
//当拉动进度条时，所有窗口的视频都会同步刷新到指定的帧数播放  
  
  
cvSetImageROI(img,cvRect(x,y,width,height));//设置ROI  
//如果ROI为NULL并且参数rect的值不等于整个图像，则ROI被分配。  
//cvRect参数分别为矩形左上角x，y坐标，矩形宽，高。  
IplImage *ImageCut = cvCreateImage(cvGetSize(img),8,3);  
cvCopy(img,ImageCut);//将原图像的ROI赋给新图像ImageCut  
/*ROI（region of interest），感兴趣区域。机器视觉、图像处理中， 
从被处理的图像以方框、圆、椭圆、不规则多边形等方式勾勒出需要处 
理的区域，称为感兴趣区域，ROI。在Halcon、OpenCV、Matlab等机器 
视觉软件上常用到各种算子（Operator）和函数来求得感兴趣区域ROI， 
并进行图像的下一步处理。在图像处理领域，感兴趣区域(ROI) 是从图 
像中选择的一个图像区域，这个区域是你的图像分析所关注的重点。圈 
定该区域以便进行进一步处理。使用ROI圈定你想读的目标，可以减少 
处理时间，增加精度。完成后可以释放ROI回到原来的视频尺寸。*/   
  
  
//创建用于反透视变换的图像  
IplImage *ImageIPM = cvCreateImage(cvGetSize(ImageCut),8,3);  
  
cvShowImage("OriginalView",ImageCut);  
cvWarpPerspective(ImageCut,ImageIPM,map_matrix);  
//对图像做反透视变换，第一个参数为原图，第二个为目标图，第三个为变换矩阵  
cvShowImage("IPMview",ImageIPM);  
  
  
  
//为了进行更精确的直线检测需要去除道路两旁的障碍，因此再次缩小ROI  
cvSetImageROI(ImageIPM,cvRect(330,0,200,256));//设置新的ROI  
  
//创建一个灰度图像  
IplImage* ImageIPM2 = cvCreateImage(cvGetSize(ImageIPM), 8, 1);  
cvCvtColor(ImageIPM,ImageIPM2,CV_BGR2GRAY);    
cvErode(  ImageIPM2,ImageIPM2, NULL,2); //腐蚀    
cvDilate( ImageIPM2,ImageIPM2, NULL,6); //膨胀   
cvShowImage("Erode&Dilate",ImageIPM2);  
  
  
IplImage *ImageCut2 = cvCreateImage(cvGetSize(ImageIPM2),8,1);  
cvCopy(ImageIPM2,ImageCut2);//将透视变换后图像的ROI赋给新图像ImageCut2  
  
//创建用于Canny变换的图像  
IplImage *img_thres = cvCreateImage(cvGetSize(ImageCut2),8,1);  
cvCanny(ImageCut2,img_thres,50,100);  
cvShowImage("AfterCanny",img_thres);  
  
cvSmooth(img_thres,img_thres,CV_GAUSSIAN,3,1,0);//高斯模糊平滑处理  
//cvShowImage("AfterSmooth",img_thres);//有的视频使用模糊处理后对直线检测更好  
  
  
/************************************* Hough ***************************************/  
/*函数说明：CvSeq* cvHoughLines2(CvArr* image,void* line_storage,int mehtod, 
double rho,double theta,int threshold,double param1 =0,double param2 =0); 
image为要做hough变换的图像，line_storage为检测到的线段存储仓， 可以是内存存储仓  
(此时，一个线段序列在存储仓中被创建，并且由函数返回），然后是hough变换的类型method 
可以是CV_HOUGH_STANDARD（标准变换），CV_HOUGH_PROBABILISTIC（概率 Hough 变换）以及 
CV_HOUGH_MULTI_SCALE（多尺度霍夫变换）。rho 与像素相关单位的距离精度 theta 弧度测量 
的角度精度 threshold 阈值参数。如果相应的累计值大于 threshold， 则函数返回这条线段. 
param1，2对标准变换无用，设为0。 1对概率 Hough 变换是最小线段长度.2对概率 Hough 变换， 
表示在同一条直线上进行碎线段连接的最大间隔值(gap), 即当同一条直线上的两条碎线段之间的 
间隔小于param2时，将其合二为一。*/  
lines = cvHoughLines2(img_thres,storage,CV_HOUGH_PROBABILISTIC,1,CV_PI/180,50,90,50);  
printf("Lines number: %d\n",lines->total);  
  
//根据Hough变换后所得线段的斜率筛选出条件合适的  
for (int i=0;i<lines->total;i++)    
{    
double k = INF;//初始化斜率为无限大  
CvPoint *line = (CvPoint *)cvGetSeqElem(lines,i);//line包含两个点line[0]和line[1]  
if(line[0].x - line[1].x != 0) k = (double)(line[0].y - line[1].y)/(double)(line[0].x - line[1].x);  
  
  
//printf("x1: %d,  y1: %d,  x2: %d,  y2: %d\n",line[0].x, line[0].y, line[1].x, line[1].y);  
//printf("k: %lf\n\n",k);  
  
  
if(k<-4.5 || k>4.5) cvLine(ImageIPM,line[0],line[1],CV_RGB(0,255,0),2,CV_AA);  
//else if(k>-1 && k<1  && lines->total>25) cvLine(ImageIPM,line[0],line[1],CV_RGB(0,255,0),2,CV_AA);  
//因为cvLine绘图只有图是3通道图时才能显示线的颜色，所以用ImageIPM作为绘线的地图  
//第二三个参数为线的起点终点，第四个为四射，第五个为线的粗细  
}    
cvShowImage("Hough",ImageIPM);  
/**********************************************************************************/  
  
  
cvResetImageROI(ImageIPM);//释放ROI  
cvWarpPerspective(ImageIPM,ImageIPM,inverse);//对效果图进行透视变换回到原视角  
  
  
//调节系数放大图像方便更清晰地浏览细节   
double fScale = 1.1;        //可调节的放大倍数，注：若fScale<0则缩小画面  
   CvSize czSize;              //目标图像尺寸    
IplImage *result = NULL;     
        
//计算目标图像大小    
czSize.width = ImageIPM->width * fScale;    
czSize.height = ImageIPM->height * fScale;    
        
//创建图像并放大    
result = cvCreateImage(czSize, ImageIPM->depth, ImageIPM->nChannels);    
cvResize(ImageIPM, result, CV_INTER_AREA);    
  
  
cvShowImage("Result", result);  
  
  
  
  
char c = cvWaitKey(33); //每隔33ms播放下一帧  
if(c == 27) break;  //按下ESC时可退出播放  
if(c == 32){  //按下空格键可暂停视频播放  
while(1){  
char c = cvWaitKey(0);  
if(c == 32) break;//再次按下则继续播放  
}  
}  
  
  
//释放使用过的图像内存  
cvReleaseImage(&ImageCut);  
cvReleaseImage(&ImageCut2);  
cvReleaseImage(&ImageIPM);  
cvReleaseImage(&img_thres);  
cvReleaseImage(&result);  
  
  
//让进度条随着视频播放滚动  
cur_frame = (int)cvGetCaptureProperty(g_capture,CV_CAP_PROP_POS_FRAMES);//提取当前帧           
        cvSetTrackbarPos("Frames","Result",cur_frame);//设置进度条位置  
  
img = cvQueryFrame(g_capture);//抓取下一帧的画面  
}  
  
  
cvReleaseCapture(&g_capture);//释放capture，同时也会释放img  
return 0;  
}  
  
  
void onTrackbarSlide(int pos){//回调函数  
if (pos!=cur_frame){  
//如果回调函数onTrackbarSlide(int pos)中当前的函数参数pos与全局变量相等，  
//说明是滚动条自动移动造成的调用，不必重新设置g_capture的当前帧  
cvSetCaptureProperty(  
g_capture,  
CV_CAP_PROP_POS_FRAMES,  
pos  
);  
}  
}  
