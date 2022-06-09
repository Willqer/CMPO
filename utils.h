//
// Created by Lech Kulesza on 30/01/2021.
//

#include <string>
#include <iostream>
#include <functional>
#include <vector>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "DisjointForest.h"

int getSingleIndex(int row, int col, int totalColumns);
int getEdgeArraySize(int rows, int columns);
std::vector<edge_pointer> setEdges(const std::vector<pixel_pointer> &pixels, std::string colorSpace, int rows, int columns);
cv::Mat addColorToSegmentation(component_struct_pointer componentStruct, int rows, int columns);
std::vector<pixel_pointer> constructImagePixels(const cv::Mat &img, int rows, int columns);
cv::Mat checkIfRectangle(component_struct_pointer componentStruct, const int rows, const int columns, float ratio);