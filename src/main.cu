#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "pgm.h"

const double degreeInc = 0.5;
const int degreeBins = static_cast<int>(180 / degreeInc);
const int rBins = 100;
const double radInc = degreeInc * M_PI / 180;

// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
    double rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2.0;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
    *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
    memset (*acc, 0, sizeof(int) * rBins * degreeBins); //init en ceros
    int xCent = w / 2;
    int yCent = h / 2;
    double rScale = 2.0 * rMax / rBins;

    for (int i = 0; i < w; i++) //por cada pixel
        for (int j = 0; j < h; j++) //...
        {
            int idx = j * w + i;
            if (pic[idx] > 0) //si pasa thresh, entonces lo marca
            {
                int xCoord = i - xCent;
                int yCoord = yCent - j;  // y-coord has to be reversed
                double theta = 0.0;         // actual angle
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
                {
                    double r = xCoord * cos (theta) + yCoord * sin (theta);
                    int rIdx = (r + rMax) / rScale;
                    if (rIdx >= 0 && rIdx < rBins)
                    {
                        (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                    }
                    theta += radInc;
                }
            }
        }
}

// GPU kernel. One thread per image pixel is spawned.
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, double rMax, double rScale, double *d_Cos, double *d_Sin) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;  // Limitar el acceso a hilos válidos

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = (gloID % w) - xCent;
    int yCoord = yCent - (gloID / w);

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            double r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            if (rIdx >= 0 && rIdx < rBins)
            {
            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
            }
        }
    }
}

int main(int argc, char **argv) {
    int i;

    if (argc < 2) {
        printf("Usage: %s <input_image.pgm>\n", argv[0]);
        return -1;
    }

    PGMImage inImg(argv[1]);
    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;
    
    double* d_Cos;
    double* d_Sin;

    cudaMalloc ((void **) &d_Cos, sizeof (double) * degreeBins);
    cudaMalloc ((void **) &d_Sin, sizeof (double) * degreeBins);

    // CPU calculation
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    double *pcCos = (double *) malloc(sizeof(double) * degreeBins);
    double *pcSin = (double *) malloc(sizeof(double) * degreeBins);
    double rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    cudaMemcpy(d_Cos, pcCos, sizeof(double) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof(double) * degreeBins, cudaMemcpyHostToDevice);

    double rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    double rScale = 2 * rMax / rBins;

    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

    h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

    cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
    cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
    cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

    // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    //1 thread por pixel
    int blockNum = ceil (w * h / 256);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel Execution Time: %f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    for (i = 0; i < degreeBins * rBins; i++)
    {
        if (cpuht[i] != h_hough[i])
        printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
    printf("Done!\n");


    const int threshold = 4240 ;

    // Crear imagen a color para dibujar las líneas
    cv::Mat img(h, w, CV_8UC1, inImg.pixels);
    cv::Mat imgColor;
    cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);

    int xCenter = (w / 2);
    int yCenter = (h / 2);

    // Vector para almacenar líneas con sus pesos
    std::vector<std::pair<cv::Vec2f, int>> linesWithWeights;

    // Recorrer el acumulador y recoger las líneas
    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            int weight = h_hough[(rIdx * degreeBins) + tIdx];

            if (weight > threshold) {  // Solo incluir líneas relevantes
                float rValue = ((rIdx * rScale) - rMax);  // Distancia r
                float theta = (tIdx * radInc);           // Ángulo theta
                linesWithWeights.push_back(std::make_pair(cv::Vec2f(theta, rValue), weight));
            }
        }
    }

    // Ordenar las líneas por peso en orden descendente
    std::sort(
        linesWithWeights.begin(),
        linesWithWeights.end(),
        [](const std::pair<cv::Vec2f, int> &a, const std::pair<cv::Vec2f, int> &b) {
            return a.second > b.second;
        }
    );

    // Dibujar las líneas principales
    for (int i = 0; i < std::min(threshold, static_cast<int>(linesWithWeights.size())); i++) {
        cv::Vec2f lineParams = linesWithWeights[i].first;
        float theta = lineParams[0];
        float r = lineParams[1];

        double cosTheta = cos(theta);
        double sinTheta = sin(theta);

        double xValue = xCenter - (r * cosTheta);
        double yValue = yCenter - (r * sinTheta);
        double alpha = 1000;  // Factor para extender las líneas

        // Dibujar la línea
        cv::line(
            imgColor,
            cv::Point(cvRound(xValue + (alpha * (-sinTheta))),
                    cvRound(yValue + (alpha * cosTheta))),
            cv::Point(cvRound(xValue - (alpha * (-sinTheta))),
                    cvRound(yValue - (alpha * cosTheta))),
            cv::Scalar(0, 185, 0),
            1,
            cv::LINE_AA
        );
    }

    // Guardar la imagen resultante
    cv::imwrite("outputGlobal.png", imgColor);
    printf("Generated marked image: output.png \n");

    cudaFree(d_in);
    cudaFree(d_hough);
    cudaFree(d_Cos);
    cudaFree(d_Sin);
    free(h_hough);
    free(pcCos);
    free(pcSin);

    return 0;
}