#include <stdio.h>
#include <stdlib.h>
#include <math.h>

unsigned char *readPGM(const char *filename, int *width, int *height, int *maxval);
void writePGM(const char *filename, unsigned char *image, int width, int height, int maxval);
unsigned char *resampleImage(unsigned char *image, int oldWidth, int oldHeight, int newWidth, int newHeight);

int main() {
    const char *inputFilename = "C:\\Users\\usaal\\Downloads\\MRI-brain.pgm";
    
    int width, height, maxval;
    float scaleFactorUp = 4.0f;   // Up-sampling factor
    float scaleFactorDown = 0.25f; // Down-sampling factor
    
    unsigned char *image = readPGM(inputFilename, &width, &height, &maxval);
    if (!image) {
        fprintf(stderr, "Error reading image. Make sure the file exists at: %s\n", inputFilename);
        return 1;
    }
    
    int newWidthUp = (int)((float)width * scaleFactorUp);
    int newHeightUp = (int)((float)height * scaleFactorUp);
    unsigned char *resampledImageUp = resampleImage(image, width, height, newWidthUp, newHeightUp);
    if (!resampledImageUp) {
        free(image);
        return 1;
    }

    int newWidthDown = (int)((float)width * scaleFactorDown);
    int newHeightDown = (int)((float)height * scaleFactorDown);
    unsigned char *resampledImageDown = resampleImage(image, width, height, newWidthDown, newHeightDown);
    if (!resampledImageDown) {
        free(image);
        free(resampledImageUp);
        return 1;
    }
    
    writePGM("MRI-brain_up.pgm", resampledImageUp, newWidthUp, newHeightUp, maxval);
    writePGM("MRI-brain_down.pgm", resampledImageDown, newWidthDown, newHeightDown, maxval);
    
    free(image);
    free(resampledImageUp);
    free(resampledImageDown);
    
    printf("Up-sampling and Down-sampling complete. Output saved as MRI-brain_up.pgm and MRI-brain_down.pgm.\n");
    return 0;
}

unsigned char *readPGM(const char *filename, int *width, int *height, int *maxval) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Error opening file");
        return NULL;
    }
    
    char magic[3];
    if (!fgets(magic, sizeof(magic), fp) || magic[0] != 'P' || magic[1] != '5') {
        fprintf(stderr, "Invalid PGM format\n");
        fclose(fp);
        return NULL;
    }

    char c;
    do {
        c = fgetc(fp);
        if (c == '#') {
            while (fgetc(fp) != '\n'); 
        }
    } while (c == '#');
    ungetc(c, fp); 

    if (fscanf(fp, "%d %d %d", width, height, maxval) != 3) {
        fprintf(stderr, "Error reading PGM header\n");
        fclose(fp);
        return NULL;
    }
    fgetc(fp);

    size_t imageSize = (size_t)(*width) * (size_t)(*height);
    unsigned char *image = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
    if (!image) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }

    if (fread(image, 1, imageSize, fp) != imageSize) {
        fprintf(stderr, "Error reading PGM image data\n");
        free(image);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return image;
}

void writePGM(const char *filename, unsigned char *image, int width, int height, int maxval) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Error opening output file");
        return;
    }
    
    fprintf(fp, "P5\n%d %d\n%d\n", width, height, maxval);
    fwrite(image, 1, (size_t)width * (size_t)height, fp);
    fclose(fp);
}

unsigned char *resampleImage(unsigned char *image, int oldWidth, int oldHeight, int newWidth, int newHeight) {
    size_t newSize = (size_t)newWidth * (size_t)newHeight;
    unsigned char *newImage = (unsigned char *)malloc(newSize * sizeof(unsigned char));
    if (!newImage) {
        fprintf(stderr, "Memory allocation failed for new image\n");
        return NULL;
    }
    
    float xRatio = ((float)(oldWidth - 1)) / ((float)(newWidth - 1));
    float yRatio = ((float)(oldHeight - 1)) / ((float)(newHeight - 1));

    for (int j = 0; j < newHeight; j++) {
        for (int i = 0; i < newWidth; i++) {
            float x = (float)i * xRatio;
            float y = (float)j * yRatio;
            
            int x1 = (int)x;
            int y1 = (int)y;
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            
            if (x2 >= oldWidth) x2 = oldWidth - 1;
            if (y2 >= oldHeight) y2 = oldHeight - 1;
            
            float xDiff = x - (float)x1;
            float yDiff = y - (float)y1;
            
            unsigned char I00 = image[y1 * oldWidth + x1];
            unsigned char I10 = image[y1 * oldWidth + x2];
            unsigned char I01 = image[y2 * oldWidth + x1];
            unsigned char I11 = image[y2 * oldWidth + x2];

            float pixelValue = (1 - xDiff) * (1 - yDiff) * I00 +
                               xDiff * (1 - yDiff) * I10 +
                               (1 - xDiff) * yDiff * I01 +
                               xDiff * yDiff * I11;
            
            newImage[j * newWidth + i] = (unsigned char)roundf(pixelValue);
        }
    }
    return newImage;
}