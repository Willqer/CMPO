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


int getSingleIndex(const int row, const int col, const int totalColumns){
    return (row*totalColumns) + col;
}

std::vector<std::string> split(const std::string& s, const char separator)
{
    std::vector<std::string> output;
    std::string::size_type prev_pos = 0, pos = 0;
    while((pos = s.find(separator, pos)) != std::string::npos)
    {
        std::string substring( s.substr(prev_pos, pos-prev_pos) );
        output.push_back(substring);
        prev_pos = ++pos;
    }
    output.push_back(s.substr(prev_pos, pos-prev_pos)); // Last word
    return output;
}

int getEdgeArraySize(const int rows,const int columns){
    int firstColumn = 3 * (rows-1);
    int lastColumn = 2 * (rows - 1);
    int middleValues = 4 * (rows - 1 ) * (columns - 2);
    int lastRow = columns - 1;
    return firstColumn + lastColumn + middleValues + lastRow;
}

std::vector<pixel_pointer> constructImagePixels(const cv::Mat &img, int rows, int columns){
    std::vector<pixel_pointer> pixels(rows*columns);

    component_pointer firstComponent = makeComponent(0, 0, img.at<cv::Vec3b>(0, 0));
    component_struct_pointer firstComponentStruct =std::make_shared<ComponentStruct>();
    firstComponentStruct->component = firstComponent;
    auto previousComponentStruct = firstComponentStruct;
    int index;

    for(int row=0; row < rows; row++)
    {
        for(int column=0; column < columns; column++)
        {
            component_pointer component=makeComponent(row, column, img.at<cv::Vec3b>(row, column));
            component_struct_pointer newComponentStruct = std::make_shared<ComponentStruct>();
            newComponentStruct->component = component;
            newComponentStruct->previousComponentStruct = previousComponentStruct;
            previousComponentStruct->nextComponentStruct = newComponentStruct;
            component->parentComponentStruct = newComponentStruct;
            previousComponentStruct = newComponentStruct;
            index = getSingleIndex(row, column, columns);
            pixels[index] = component->pixels.at(0);
        }
    }
    firstComponentStruct = firstComponentStruct->nextComponentStruct;
    firstComponentStruct->previousComponentStruct = nullptr;
    return pixels;
}

std::vector<edge_pointer> setEdges(const std::vector<pixel_pointer> &pixels, const std::string colorSpace, const int rows, const int columns){
    int edgeArraySize = getEdgeArraySize(rows, columns);
    std::vector<edge_pointer> edges(edgeArraySize);
    std::function<double(pixel_pointer, pixel_pointer)> edgeDifferenceFunction;
    if (colorSpace == "rgb"){
        edgeDifferenceFunction = rgbPixelDifference;
    }else{
        edgeDifferenceFunction = grayPixelDifference;
    }
    int edgeCount = 0;
    for(int row=0; row < rows; ++row){
        for(int column=0; column < columns; ++column) {
            pixel_pointer presentPixel = pixels[getSingleIndex(row, column, columns)];
            if(row < rows - 1){
                if(column == 0){
                    edges[edgeCount++] = createEdge(presentPixel, pixels[getSingleIndex(row, column+1, columns)], edgeDifferenceFunction);
                    edges[edgeCount++] = createEdge(presentPixel, pixels[getSingleIndex(row+1, column+1, columns)], edgeDifferenceFunction);
                    edges[edgeCount++] = createEdge(presentPixel , pixels[getSingleIndex(row+1, column, columns)], edgeDifferenceFunction);
                }
                else if(column==columns-1){
                    edges[edgeCount++] = createEdge(presentPixel, pixels[getSingleIndex(row+1, column, columns)], edgeDifferenceFunction);
                    edges[edgeCount++] = createEdge(presentPixel, pixels[getSingleIndex(row+1, column-1, columns)], edgeDifferenceFunction);
                }else{
                    edges[edgeCount++] = createEdge(presentPixel, pixels[getSingleIndex(row, column+1, columns)], edgeDifferenceFunction);
                    edges[edgeCount++] = createEdge(presentPixel, pixels[getSingleIndex(row+1, column+1, columns)], edgeDifferenceFunction);
                    edges[edgeCount++] = createEdge(presentPixel, pixels[getSingleIndex(row+1, column, columns)], edgeDifferenceFunction);
                    edges[edgeCount++] = createEdge(presentPixel, pixels[getSingleIndex(row+1, column-1, columns)], edgeDifferenceFunction);
                }
            }
            else if(row == rows - 1){
                if(column != columns - 1) {
                    edges[edgeCount++] = createEdge(presentPixel, pixels[getSingleIndex(row,column+1, columns)], edgeDifferenceFunction);
                }
            }
        }
    }
    return edges;
}

int getRandomNumber(const int min,const int max)
{
    //from learncpp.com
    static constexpr double fraction { 1.0 / (RAND_MAX + 1.0) };
    return min + static_cast<int>((max - min + 1) * (std::rand() * fraction));
}

cv::Mat addColorToSegmentation(component_struct_pointer componentStruct, const int rows, const int columns){
    cv::Mat segmentedImage(rows, columns, CV_8UC3);
    int r = 0;
    int b = 0;
    int g = 0;
    do{
        cv::Vec3b pixelColor= {static_cast<unsigned char>(b) ,static_cast<unsigned char>(g) ,static_cast<unsigned char>(r)};
        for(const auto& pixel: componentStruct->component->pixels){
            segmentedImage.at<cv::Vec3b>(cv::Point(pixel->column,pixel->row)) = pixelColor;
        }
        r += 10;
        g += 10;
        b += 10;
        componentStruct = componentStruct->nextComponentStruct;
    }while(componentStruct);

    return segmentedImage;
}

cv::Mat checkIfRectangle(component_struct_pointer componentStruct, const int rows, const int columns, float ratio) {
    cv::Mat segmentedImage(rows, columns, CV_8UC3);
    int r = 0;
    int b = 0;
    int g = 0;
    int i = 0;
    do{
        int j = 0;
        for(const auto& pixel: componentStruct->component->pixels){
            cv::Point firstPoint;
            cv::Point lastPoint;
            if(j == 0)
                firstPoint = cv::Point(pixel->column,pixel->row);
            if(j == componentStruct->component->pixels.size()-1)
                lastPoint = cv::Point(pixel->column, pixel->row);
            j++;
        }
        componentStruct = componentStruct->nextComponentStruct;
        i++;
        r += 10;
        g += 10;
        b += 10;
    } while (componentStruct);

    return segmentedImage;
}