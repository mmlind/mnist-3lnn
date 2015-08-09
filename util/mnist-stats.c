/**
 * @file mnist-stats.c
 * @brief Utitlies for displaying processing details in the terminal
 * @author Matt Lind
 * @date July 2015
 */

#include <string.h>

#include "screen.h"
#include "mnist-utils.h"
#include "mnist-stats.h"




/**
 * @details Outputs a 28x28 text frame at a defined screen position
 */

void displayImageFrame(int row, int col){
    
    if (col!=0 && row!=0) locateCursor(row, col);

    printf("------------------------------\n");
    
    for (int i=0; i<MNIST_IMG_HEIGHT; i++){
        for (int o=0; o<col-1; o++) printf(" ");
        printf("|                            |\n");
    }

    for (int o=0; o<col-1; o++) printf(" ");
    printf("------------------------------\n");
    
}




/**
 * @details Outputs a 28x28 MNIST image as charachters ("."s and "X"s)
 */

void displayImage(MNIST_Image *img, int lbl, int cls, int row, int col){

    char imgStr[(MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH)+((col+1)*MNIST_IMG_HEIGHT)+1];
    strcpy(imgStr, "");
    
    for (int y=0; y<MNIST_IMG_HEIGHT; y++){
        
        for (int o=0; o<col-2; o++) strcat(imgStr," ");
//        strcat(imgStr,"|");
        
        for (int x=0; x<MNIST_IMG_WIDTH; x++){
            strcat(imgStr, img->pixel[y*MNIST_IMG_HEIGHT+x] ? "X" : "." );
        }
        strcat(imgStr,"\n");
    }
    
    if (col!=0 && row!=0) locateCursor(row, 0);
    printf("%s",imgStr);
    
    printf("     Label:%d   Classification:%d\n\n",lbl,cls);
    
}




/**
 * @details Outputs reading progress while processing MNIST training images
 */

void displayTrainingProgress(int imgCount, int errCount, int y, int x){
    
    double progress = (double)(imgCount+1)/(double)(MNIST_MAX_TRAINING_IMAGES)*100;
    
    if (x!=0 && y!=0) locateCursor(y, x);
    
    printf("1: TRAINING: Reading image No. %5d of %5d images [%3d%%]  ",(imgCount+1),MNIST_MAX_TRAINING_IMAGES,(int)progress);

    
    double accuracy = 1 - ((double)errCount/(double)(imgCount+1));
    
    printf("Result: Correct=%5d  Incorrect=%5d  Accuracy=%5.4f%% \n",imgCount+1-errCount, errCount, accuracy*100);
    
}



void displayTestingProgress(int imgCount, int errCount, int y, int x){
    
    double progress = (double)(imgCount+1)/(double)(MNIST_MAX_TESTING_IMAGES)*100;
    
    if (x!=0 && y!=0) locateCursor(y, x);
    
    printf("2: TESTING:  Reading image No. %5d of %5d images [%3d%%]  ",(imgCount+1),MNIST_MAX_TESTING_IMAGES,(int)progress);
    
    
    double accuracy = 1 - ((double)errCount/(double)(imgCount+1));
    
    printf("Result: Correct=%5d  Incorrect=%5d  Accuracy=%5.4f%% \n",imgCount+1-errCount, errCount, accuracy*100);
    
}





