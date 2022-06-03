#include "stdafx.h"
#include "common.h"

#define DMAX 75
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage) {
	int val = (int)(percentage * 100) + 1;
	int lpad = (int)(percentage * PBWIDTH);
	int rpad = PBWIDTH - lpad;
	printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
	fflush(stdout);
}
bool isInside(Mat img, Point p) {
	return (p.y >= 0 && p.x >= 0 && p.y < img.rows&& p.x < img.cols);
}
bool isInside(int rows, int cols, Point p) {
	return (p.y >= 0 && p.x >= 0 && p.y < rows&& p.x < cols);
}

//int sumAggregateDifferences(Mat left, Mat right, Point p_left, Point p_right) {
//	int sum = 0;
//	for (int i = -3; i <= 3; ++i) {
//		for (int j = -3; j <= 3; ++j) {
//			Point dir = Point(j, i);
//			Point pl = p_left + dir;
//			Point pr = p_right + dir;
//			if (isInside(left, pl)) {
//				int lval = isInside(left, pl) ? left.at<uchar>(pl) : 0;
//				int rval = isInside(right, pr) ? right.at<uchar>(pr) : 0;
//				int temp = abs(lval - rval);
//				sum += temp;
//			}
//		}
//	}
//	return sum;
//}

void calculateCosts(Mat left, Mat right, int*** cost) {
#pragma omp parallel for
	for (int i = 0; i < left.rows; ++i) {
		if (i % (left.rows / 100) == 0)
		{
			printProgress(((float)i / left.rows));
		}
#pragma omp parallel for
		for (int j = 0; j < left.cols; ++j) {
			Point leftP = Point(j, i);
#pragma omp parallel for
			for (int d = 0; d <= DMAX; ++d) {
				Point rightP = leftP - Point(d, 0);
				if (isInside(right, rightP)) {
					int sum = 0;
#pragma omp parallel for
					for (int i = -3; i <= 3; ++i) {
#pragma omp parallel for
						for (int j = -3; j <= 3; ++j) {
							Point dir = Point(j, i);
							Point pl = leftP + dir;
							Point pr = rightP + dir;
							if (isInside(left, pl)) {
								int lval = isInside(left, pl) ? left.at<uchar>(pl) : 0;
								int rval = isInside(right, pr) ? right.at<uchar>(pr) : 0;
								int temp = abs(lval - rval);
								sum += temp;
							}
						}
					}
					cost[leftP.y][leftP.x][d] = sum;
				}
				else {
					d = DMAX + 1;
				}
			}
		}
	}
}
Mat computeDisparity(int*** cost, int rows, int cols)
{
	Mat disparityMap = Mat(rows, cols, CV_8UC1);
	disparityMap.setTo(0);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int minD = INT16_MAX;
			int disparity = -5;
			for (int d = 0; d < DMAX; d++) {
				if (cost[i][j][d] < minD) {
					minD = cost[i][j][d];
					disparity = d;
				}
			}
			disparityMap.at<uchar>(i, j) = disparity;
		}
	}
	double range = 256. / DMAX;
	disparityMap *= range;
	return disparityMap;
}

int main() {//image not displaying means going out of bounds
	clock_t begin = clock();
	char filename[] = "bike.png";
	//Mat left = imread("Images/Motorcycle/imL.png", cv::IMREAD_GRAYSCALE);
	//Mat right = imread("Images/Motorcycle/imR.png", cv::IMREAD_GRAYSCALE);
	Mat left = imread("Images/Bike/imL.png", cv::IMREAD_GRAYSCALE);
	Mat right = imread("Images/Bike/imR.png", cv::IMREAD_GRAYSCALE);
	printf("Allocating memory...\n");
	int*** cost = new int** [left.rows];
	for (int i = 0; i < left.rows; i++) {
		cost[i] = new int* [left.cols]();
		for (int j = 0; j < left.cols; j++) {
			cost[i][j] = new int[DMAX]();
			for (int d = 0; d < DMAX + 1; d++) {
				cost[i][j][d] = INT_MAX;
			}
		}
	}
	int*** S = new int** [left.rows];
	for (int i = 0; i < left.rows; i++) {
		S[i] = new int* [left.cols]();
		for (int j = 0; j < left.cols; j++) {
			S[i][j] = new int[DMAX]();
			for (int d = 0; d < DMAX + 1; d++) {
				S[i][j][d] = INT_MAX;
			}
		}
	}
	printf("Calculating Costs...\n");
	calculateCosts(left, right, cost);


	printf("Aggregating...\n");
	int P1 = 3;
	int P2 = 20;
	Point dirs[8] = { {1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,1},{1,1} };

	printf("aggregate_costs:main computations...\n");
	for (int i = 0; i < left.rows; i++) {
		for (int j = 0; j < left.cols; j++) {
			for (int d = 0; d < DMAX; d++) {
				//check that the depth is valid
				if (j >= d) {
					for (int dir = 0; dir < 8; dir++) {
						int l = 0;
						l += cost[i][j][d];

						int y = dirs[dir].y;
						int x = dirs[dir].x;
						Point p = Point(j - y, i - x);
						if (isInside(left.rows, left.cols, p)) {
							int a = cost[i - x][j - y][d];//Lr(p-r,d)
							int b = (d > 0) ? (cost[i - x][j - y][d - 1] + P1) : P1;//Lr(p-r,d-1)+P1
							int c = cost[i - x][j - y][d + 1] + P1;//Lr(p-r,d+1)+P1
							//min of all depths from prev pixel
							int minD = INT16_MAX;//min(Lr(p-r,i)+P2)
							for (int d1 = 0; d1 < DMAX; d1++) {
								int z = cost[i - x][j - y][d1] + P2;
								minD = min(z, minD);
							}
							int temp = min(min(a, b), min(c, minD)) - (minD - P2);//formula 13
							l += temp;
						}
						else {
							S[i][j][d] += l;
							break;
						}
						S[i][j][d] += l;//S is the sum of all path costs
					}
				}
			}
		}
	}

	printf("Computing disparity\n");
	Mat disparityMap = computeDisparity(S, left.rows, left.cols);
	//imshow("res", disparityMap);
	imwrite(filename, disparityMap);

	clock_t end = clock();
	double time = double(end - begin) / CLOCKS_PER_SEC;
	printf("Done in %.2lf minutes.\n", time / 60.);
	waitKey(0);
	return 0;
}