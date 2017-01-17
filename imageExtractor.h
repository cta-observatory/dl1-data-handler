#ifndef __IMAGEEXTRACTOR_H__  
#define __IMAGEEXTRACTOR_H__  

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath> 
#include <algorithm> 
#include <iomanip>
#include <cassert>

#include <sys/stat.h>
#include <unistd.h>

#include <TH2F.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TCanvas.h>
#include <TPaveText.h>
#include <TStyle.h>
#include <TMath.h>
#include <TError.h>
#include <TSystem.h>
#include <TUnixSystem.h>

#include <highgui.hpp>
#include <exiv2.hpp>
#include <xmp.hpp>
#include <image.hpp>
#include <properties.hpp>

struct metadata
{
    string eventType;
    float impactParameter;
    unsigned int eventID;
    unsigned int telNum;
    UShort_t MCprim;
    float MCe0;
    float MCxcore;
    float MCycore;
    float telx;
    float tely;
    float MCze;
    float MCaz;
    float MCxoff;
    float MCyoff;
    int pedrms;
};

int processCAREdata(TFile *file, TPaveText *pt, TH2F *hcamera, TCanvas *ccamera, vector<double> v_xcoord, vector<double> v_ycoord, int num_channels, string output_dir)

int processEDdata(TFile *file, TPaveText *pt, TH2F *hcamera, TCanvas *ccamera, vector<double> v_xcoord, vector<double> v_ycoord, int num_channels, string output_dir)

int getFirstBin(unsigned short int trace[][][], int tel, int channel, int num_samples, int ped_rm)

void readConfig(string filepath, vector<int>& channels, vector<double>& x_coord, vector<double>& y_coord)

void createImage(TH2F *hcam, string img_name,struct metadata md)

#endif 
