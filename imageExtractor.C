///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// imageExtractor.C                                                          //
//                                                                           // 
// Read simulated CTA data in DST format and generates events images in      //
// eps and png formats.                                                      //
//                                                                           //
// Author: Daniel Nieto (nieto@nevis.columbia.edu)                           //
//                                                                           //
// Additional work by: Bryan Kim (bryan.sanghyuk.kim@gmail.com)              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

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

#include "imageExtractor.h"

bool print_eps = false;
bool debug = false;

int main(int argc, char** argv)
{
    /*
     * Argument and option parsing
     */

    //gErrorIgnoreLevel = 5000;
    std::string data_file;
    std::string config_file;
    std::string output_dir = "./";
    std::string format = "ed";

    std::string help =  "Usage is -i <input file> -c <camera file> -f <data format> -o <output directory> -e -d\n"
    "-i and -c fields are mandatory\n"
    "-o flag and field are optional, defaults to $pwd\n"
    "-f flag and field are optional, defaults to \"ed\". Use ed for eventdisplay, care for CARE format\n"
    "-e flag is optional, generates eps files in addition to png images\n"
    "-d flag is optional, prints debug output during execution, false by default\n"
    "-h option for help\n";

    int opt;
    while((opt = getopt(argc,argv,"i:c:o:f:ehd")) != -1)
    {
        switch(opt)
        {
            case 'i':
                data_file = optarg;
                break;
            case 'o':
                output_dir = optarg;
                break;
            case 'c':
                config_file = optarg;
                break;
            case 'f':
                format = optarg;
                break;
            case 'e':
                print_eps = true;
                break;
            case 'd':
                debug = true;
                break;
            case 'h':
                std::cout << help;
                return 0;
            case '?':
                if (optopt == 'o')
                {
                    outputdir = "./";
                }
                else if (optopt == 'f')
                {
                    format = "ed";
                }
                else
                {
                    std::cerr << "Error: invalid arguments.\n" << endl;
                    std::cerr << help;
                    return 1;
                }
                break;
        }
    }
    
    std::cout << "Data file: " << data_file << endl;
    std::cout << "Camera file: " << config_file << endl;
    std::cout << "Output directory: " << (output_dir == "./" ? "Default (\"./\")" : output_dir) << endl;
    std::cout << "Generate eps files: " << (print_eps ? "Yes" : "No") << endl;
    std::cout << "Debug mode: " <<  (debug ? "Yes" : "No") << endl;

    /* 
     * Open data and config files
     */
    
    TFile::TFile *file = TFile::Open(datafile.data());
    if (f == NULL)
    {
        std::cerr << "Error: cannot open data file " << data_file.data() << endl;
        return 1;
    }

    struct stat s;
    if (stat(config_file.c_str(), &s) != 0) 
    {
	std::cerr << "Error: cannot open configuration file " << config_file.data() << endl;
    	return 1;
    }
  
    /*
     * Read configuration file and fill pixel (channel) number and corresponding coordinate vectors
     */

    std::vector<int> v_channels;
    std::vector<double> v_xcoord;
    std::vector<double> v_ycoord;

    readconfig(config_file,v_channels,v_xcoord,v_ycoord);

    //Square pixels assumed
    int num_channels = v_channels.size();
    double px_pitch = std::max(abs(v_xcoord.at(0)-v_xcoord.at(1)),abs(v_ycoord.at(0)-v_ycoord.at(1))); 
    double x_min = *std::min_element(v_xcoord.begin(),v_xcoord.end());
    double x_max = *std::max_element(v_xcoord.begin(),v_xcoord.end());
    double y_min = *std::min_element(v_ycoord.begin(),v_ycoord.end());
    double y_max = *std::max_element(v_ycoord.begin(),v_ycoord.end());
    double x_max_px = (x_max+px_pitch/2)/px_pitch;
    double x_min_px = (x_min-px_pitch/2)/px_pitch;
    double y_max_px = (y_max+px_pitch/2)/px_pitch;
    double y_min_px = (y_min-px_pitch/2)/px_pitch;//are these variables necessary? they are only used for debug output

    if (debug)
    {
        std::cout << "Number of channels read: " << num_channels << endl;
        std::cout << "Pixel pitch: "<< px_pitch << " mm" << endl;
        std::cout << "Camera max/min length (x): (" << x_min << " mm, " << x_max << " mm) (" << x_min_px << " px,  " << x_max_px << " px)" << endl; 
        std::cout << "Camera max/min width (y): (" << y_min << " mm, " << y_max << " mm) (" << y_min_px << " px, " << y_max_px << " px)" << endl; 
    }

    /*
     * Generate camera histogram, canvas, and text box
     */

    //should change to use calculated values, not constants
    const double AUX_PX_PITCH = 54/8;
    const int X_NUM_PX = 15*8;
    const int Y_NUM_PX = 15*8;

    TH2F::TH2F *hcamera = new TH2F::TH2F("hcamera","",X_NUM_PX,-(AUX_PX_PITCH*X_NUM_PX)/2,(AUX_PX_PITCH*X_NUM_PX)/2,Y_NUM_PX,-(AUX_PX_PITCH*Y_NUM_PX)/2,(AUX_PX_PITCH*Y_NUM_PX)/2);
    hcamera->GetXaxis()->SetTitle("X [mm]");
    hcamera->GetYaxis()->SetTitle("Y [mm]");
    hcamera->SetStats(0);

    TPaveText::TPaveText *pt = new TPaveText::TPaveText(0,0.9,1,0.99,"NDC"); //should this be stack-allocated?
    pt->SetFillStyle(0);
    pt->SetBorderSize(0); 
    
    TCanvas::TCanvas *ccamera = new TCanvas::TCanvas("ccamera","SCT Camera",500,500); 
    gStyle->SetPalette(51);
    gStyle->SetNumberContours(999);
 
    //process data in CARE format 
    if (format.compare("care") == 0)
    {
        return processCAREdata(file, pt, hcamera, ccamera, num_channels, output_dir);
    }

    //process data in evndisp format
    else if (format.compare("ed") == 0)
    {
        return processEDdata();               
    }
  
    else
    {
        std::cout << "Data format not recognized. Exiting..." << endl;
        return 1;
    }
}

int processCAREdata(TFile *file, TPaveText *pt, TH2F *hcamera, TCanvas *ccamera, vector<double> v_xcoord, vector<double> v_ycoord, int num_channels, string output_dir)
{
    std::cout << "Processing CARE data" << endl;

    TTree::TTree* data_tree = (TTree::TTree*) file->Get("Events/T0;19");//should this be left hardcoded?    
    TTree::TTree* info_tree = (TTree::TTree*) file->Get("Events/tSimulatedEvents");
    
    const int TEL_NUM = 0; //why is this value not updated?
    const int MIN_ENERGY = 0;

    int start_entry = 0;
    int stop_entry = info_tree->GetEntries();

    int ped_rm = 0;

    std::vector<bool> *trigger_bit = 0;
    float energy, x_core, y_core;
    ULong64_t event_num;
    UShort_t prim; 

    info_tree->SetBranchAddress("vTelescopeTriggerBits", &trigger_bit);
    info_tree->SetBranchAddress("energy", &energy);
    info_tree->SetBranchAddress("xcore", &x_core);
    info_tree->SetBranchAddress("ycore", &y_core);
    info_tree->SetBranchAddress("eventNumber", &event_num);
    info_tree->SetBranchAddress("MCprim", &prim);

    for (int i = start_entry; i < stop_entry+1; i++)
    {
        info_tree->GetEntry(i);

        std::string event_type;
        switch(prim)
        {
            case 0 :
                event_type = "gamma";
            case 101 :
                event_type = "proton";
        }

        char buffer[1000];

        if (debug) std::cout << "Trigger bit: " << trigger_bit->at(TEL_NUM) << endl; //tel_num == 0?

        if (trigger_bit->at(TEL_NUM) && (energy > MIN_ENERGY))
        {
            for (int j = 0; j < num_channels; j++) 
            {
                sprintf(buffer, "vFADCTraces%d",j);
              
                if (debug) std::cout << "Entry: " << i << " Branch: " << buffer << endl;

                std::vector<int> *trace = new std::vector<int>;//allocate on stack?
                TBranch::TBranch *trace_branch = new TBranch::TBranch(); 

                data_tree->SetBranchAddress(buffer, &trace, &trace_branch);
                trace_branch->GetEntry(i);
                
                int charge = 0;
                for (int k = 0; k < int(trace->size()); k++)
                {
                    charge += trace->at(k)-ped_rm;
                }

                hcamera->SetBinContent(hcamera->FindBin(v_xcoord[j],v_ycoord[j]),charge);

                if (debug) std::cout << "Total charge: " << charge << endl;
            } 

            hcamera->Draw("colz");
            pt->Clear();
            sprintf(buffer,"Event: %s",event_type.data());
            pt->AddText(buffer);
            sprintf(buffer,"Energy: %.3f TeV",energy);
            pt->AddText(buffer);
            sprintf(buffer,"Impact: %.0f m",sqrt(x_core*x_core+y_core*y_core));
            pt->AddText(buffer);
            sprintf(buffer,"ID: %llu",event_num);
            pt->AddText(buffer);
            pt->Draw();
            ccamera->Update();
            ccamera->cd();

            float impact = sqrt(x_core*x_core+y_core*y_core);

            sprintf(buffer,"%s/%u_%.3fTeV_%.0fm_T%u",output_dir.c_str(),event_num,energy,impact,TEL_NUM);
            std::string image_name = std::string(buffer);

            //collect and set metadata values
            struct metadata md;

            md.eventType = event_type.data();
            md.impactParameter = impact;
            md.eventID = event_num;
            md.telNum = ltrig_list[l];
            md.MCprim = prim;
            md.MCe0 = energy;
            md.MCxcore = x_core;
            md.MCycore = y_core;
            md.telx = pos_map_x[ltrig_list[l]];
            md.tely = pos_map_y[ltrig_list[l]];
            md.MCze = ze;
            md.MCaz = az;
            md.MCxoff = x_off;
            md.MCyoff = y_off;
            md.pedrms = ped_rm;

            if (!createimage(hcamera,event_num,md,debug)) return -1;
            ccamera->WaitPrimitive();
          
            if (printeps)
            {
                sprintf(buffer,"%s/%llu.eps",output_dir.c_str(),event_num);
                ccamera->SaveAs(buffer);
            }
            hcamera->Reset();
        }      
        std::cout << "\r" << i << "/" << stop_entry << " (" << float(i)/float(stop_entry)*100 << "%)" << flush;
    }
    std::cout << "\nDone!" << endl;
    return 0;
}

int processEDdata(TFile *file, TPaveText *pt, TH2F *hcamera, TCanvas *ccamera, vector<double> v_xcoord, vector<double> v_ycoord, int num_channels, string output_dir)
{
    std::cout << "Processing Eventdisplay data" << endl;

    TTree::TTree *data_tree = (TTree::TTree*) file->Get("dst");
    data_tree->Draw("ped>>hped","","goff");
    TH1F::TH1F *hped = (TH1F::TH1F*)gDirectory->Get("hped");

    const int MAX_TEL = 8;
    const int MAX_SAMPLES = 64;
    const int MAX_CHANNELS = 12000;
  
    //get number of telescopes from first branch of data tree
    UInt_t ntel;
    data_tree->SetBranchAddress("ntel", &ntel);
    data_tree->GetEntry(0);

    //set number of samples from first telescope
    UShort_t num_samples[ntel];
    data_tree->SetBranchAddress("numSamples", &num_samples);
    data_tree->GetEntry(0);

    //use telconfig file to generate telescope ID map and position map
    TTree::TTree* tel_tree = (TTree::TTree*) file->Get("telconfig");
    float tel_x, tel_y;
    int tel_id;
    std::map<int, int> tel_map;
    std::map<int, float> pos_map_x;
    std::map<int, float> pos_map_y;

    tel_tree->SetBranchAddress("TelX",&tel_x);
    tel_tree->SetBranchAddress("TelY",&tel_y);
    tel_tree->SetBranchAddress("TelID",&tel_id);

    for (int i = 0; i < tel_tree->GetEntries(); i++)
    {
        tel_tree->GetEntry(i);

        tel_map[tel_id]=i;
        pos_map_x[tel_id]=tel_x;
        pos_map_y[tel_id]=tel_y;

        if (debug) std::cout << "(tel id =" << tel_id << ",tel x =" << tel_x << ",tel y =" << tel_y << ")" << endl;
    }

    //check if trace array bounds are reasonable values
    std::string s = data_tree->GetBranch("Trace")->GetTitle();
    int samples = std::atoi(s.substr(s.find("][")+2,s.find_last_of("[")-s.find("][")-3).c_str());
    if (samples > MAX_SAMPLES)
    {
        std::cout << "FADC samples in data " << samples << " exceeds maximum of " << MAX_SAMPLES << endl;
        std::cout << "Quiting..." << endl;
        return 1;
    }

    int channels = std::atoi(strace.substr(s.find_last_of("[")+1,s.find_last_of("]")-s.find_last_of("[")-1).c_str());
    if (channels > MAX_CHANNELS)
    {
        std::cout << "Channels in data " << channels << " exceeds maximum of " << MAX_CHANNELS << endl;
        std::cout << "Quiting..." << endl;
        return 1;
    }

    unsigned short int trace[ntel][samples][channels] = {0,0,0};

    if (debug)
    {
        std::cout << "# of telescopes = " << ntel << ", Samples = " << samples << " # of pixels = " << channels << endl;
        std::cout << "Samples" << endl;
        for (UInt_t  i = 0; i < ntel; i++)
        {   
            std::cout << num_samples[i] << endl;
        }
    }

    UInt_t event_num = 0;
    UInt_t ltrig_list[MAX_TEL];
    UInt_t num_trig = 0;
    UInt_t ntel_data;
    UInt_t tel_data[MAX_TEL];
    UShort_t prim = 0;    
    float energy, x_core, y_core, x_off, y_off, az, ze;

    int ped_rm = hped->GetMean()-2*hped->GetRMS();

    int start_entry = 0;
    int stop_entry = data_tree->GetEntries();

    datatree->SetBranchAddress("MCe0", &energy);
    datatree->SetBranchAddress("MCxcore", &x_core);
    datatree->SetBranchAddress("MCycore", &y_core);
    datatree->SetBranchAddress("eventNumber", &event_num);
    datatree->SetBranchAddress("ntel", &ntel);
    datatree->SetBranchAddress("ntel_data", &ntel_data);
    datatree->SetBranchAddress("tel_data", &tel_data);
    datatree->SetBranchAddress("Trace", trace);
    datatree->SetBranchAddress("ntrig", &ntrig);
    datatree->SetBranchAddress("ltrig_list",ltrig_list);
    datatree->SetBranchAddress("MCprim", &prim);
    datatree->SetBranchAddress("MCxoff", &x_off);
    datatree->SetBranchAddress("MCyoff", &y_off);
    datatree->SetBranchAddress("MCze", &ze);
    datatree->SetBranchAddress("MCaz", &az);

    for (int i = start_entry; i < stop_entry; i++)
    {
        datatree->GetEntry(i);
      
        if (debug)
        {
            std::cout << i << " " << ntel << " " << ntel_data << " " << tel_data[0] << " " << energy << " " << x_core << " " << y_core << " " << event_num << endl;
            std::cout << "pedrm =" << ped_rm << endl;
            std::cout << "MCprim =" << prim << endl;
            std::cout << "ntrig =" << ntrig << endl;
            std::cout << "ltrig_list" << endl;
            for (UInt_t j = 0; j < MAX_TEL;j++)
            {
                std::cout << ltrig_list[j] << endl;
            }
            std::cout << "tel map" << endl;
            for (std::map<int,int>::iterator it=tel_map.begin(); it!=tel_map.end(); ++it)
            {
                std::cout << it->first << " => " << it->second << endl;
            }
        }
    
        for (int l = 0; l < (int)ntrig; l++)
        {
            if(debug)
            {
                std::cout << "i="<< i << " l=" << l << " ltrig_list[l]= " << ltrig_list[l] << "tel_map[ltrig_list[l]]=" << tel_map[ltrig_list[l]] << endl;
                std::cout << "num_samples[l] =" << num_samples[l] << endl;
            }

            for (int j = 0 ; j < channels; j++) 
            {
                int first_bin = getFirstBin(trace, l, j, samples, ped_rm)

                int charge = 0;
                for (int k = first_bin; k < (first_bin + 6); k++)
                {
                    charge += trace[l][k][j]-ped_rm; 
                }

                hcamera->SetBinContent(hcamera->FindBin(v_xcoord[j],v_ycoord[j]),charge);
            }

            float impact = sqrt((x_core-pos_map_x[ltrig_list[l]])*(x_core-pos_map_x[ltrig_list[l]])+(y_core-pos_map_y[ltrig_list[l]])*(y_core-pos_map_y[ltrig_list[l]]));
        
            //determine particle type based on MCprim value 
            std::string event_type;
            switch(prim)
            {
                case 0 :
                    event_type = "gamma";
                    break;
                case 101 : 
                    event_type = "proton";
                    break;
            }

            char buffer[1000];

            hcamera->Draw("colz");
            pt->Clear();
            sprintf(buffer,"Event: %s",event_type.data());
            pt->AddText(buffer);
            sprintf(buffer,"Energy: %.3f TeV",energy);
            pt->AddText(buffer);
            sprintf(buffer,"Impact: %.0f m",impact);
            pt->AddText(buffer);
            sprintf(buffer,"ID: %u",event_num);
            pt->AddText(buffer);
            sprintf(buffer,"Tel ID: %u",ltrig_list[l]);
            pt->AddText(buffer);
            pt->Draw();
            ccamera->Update();
            ccamera->cd();
            ccamera->WaitPrimitive();

            sprintf(buffer,"%s/%u_%.3fTeV_%.0fm_T%u",output_dir.c_str(),event_num,energy,impact,ltrig_list[l]);
            std::string image_name = std::string(buffer);

            //collect and set metadata values
            struct metadata md;

            md.eventType = event_type.data();
            md.impactParameter = impact;
            md.eventID = event_num;
            md.telNum = ltrig_list[l];
            md.MCprim = prim;
            md.MCe0 = energy;
            md.MCxcore = x_core;
            md.MCycore = y_core;
            md.telx = pos_map_x[ltrig_list[l]];
            md.tely = pos_map_y[ltrig_list[l]];
            md.MCze = ze;
            md.MCaz = az;
            md.MCxoff = x_off;
            md.MCyoff = y_off;
            md.pedrms = ped_rm;

            if (!createimage(hcamera,image_name,md,debug)) return 1;
            if (printeps)
            {
                sprintf(buffer,"%s.eps",buffer);
                ccamera->SaveAs(buffer);
            }
            hcamera->Reset();
        }  
        std::cout << "\r" << i << "/" << stop_entry << " ("<<int(float(i)/float(stop_entry)*100)<<"%)" << flush;
    }
    std::cout << "\nDone!"<< endl;  
    return 0;
}

int getFirstBin(unsigned short int trace[][][], int tel, int channel, int num_samples, int ped_rm)
{
    int charge = 0;    
    int max_charge = 0;      
    int first_HM_bin = 0;

    //find max charge
    for (int i = 0; i < num_samples; i++)
    {
        charge = trace[tel][i][channel]-ped_rm;
        if (charge > max_charge)
        {
            maxCharge=charge;
        }
    }

    //find first bin with charge > maxcharge/2
    for (int i = 0; i < num_samples; i++)
    {
        charge = trace[tel][i][channel]-ped_rm;
        if (charge > (max_charge/2))
        {
            first_HM_bin = i;
            break;
        }
    }

    //check if the range of integration goes out of bounds
    //if so, sum over last 6 bins

    int first_bin;
    if(firstHMbin + 5 >= num_samples)
    {
        first_bin = num_samples - 6;
    }
    else
    {
        first_bin = first_HM_bin;
    }

    return first_bin;
}

void readConfig(string filepath, vector<int>& channels, vector<double>& x_coord, vector<double>& y_coord)
{
    if (debug) std::cout << "Reading configuration file..." << endl;

    std::ifstream config_file(filepath.data());
    std::string line;
    std::string c;  //used to ignore fields
    int tel_num, tel_type, num_px;
    double x, y;

    while (getline(config_file, line))
    {
        if (line.find("* PMPIX")==0)
        {
            std::istringstream iss(line);
            iss >> c >> c >> tel_num >> tel_type >> num_px >> x >> y;
            channels.push_back(num_px);
            x_coord.push_back(x);
            y_coord.push_back(y);
        }
    }
    if (debug) std::cout << "Done" << endl;
}


void createImage(TH2F *hcam, string img_name,struct metadata md)
{
    const int IMAGE_SCALE = 1;
    const int IMAGE_DEPTH = 16;
    const int IMAGE_TYPE = CV16UC1; //CV_16UC1: 16-bit depth, unsigned int, 1 channel
    const int COMPRESSION_LEVEL = 9; //0 to 9, 9 being highest compression

    int img_x_px = hcam->GetXaxis()->GetNbins();
    int img_y_px = hcam->GetYaxis()->GetNbins();

    cv::Mat img(img_x_px,img_y_px,IMAGE_TYPE, Scalar(0));

    std::cout << "Creating PNG image..." << endl;
    std::cout << "X: "<< img_x_px << endl;
    std::cout << "Y: " << img_y_px << endl;
    std::cout << "Max. pixel value: " << hcam->GetMaximum() << endl;
    std::cout << "Max. depth: "<< std::ceil(TMath::Log2(hcam->GetMaximum())) << endl;

    if (ceil(TMath::Log2(hcam->GetMaximum())) > IMAGE_DEPTH)
    { 
        std::cerr << "Image is being truncated: increase color depth" << endl;
    }  

    for (int i = 0; i < im_x_px; i++)
    {
        for (int j = 0; j < im_y_px; j++)
        {
	    int val = std::abs(hcam->GetBinContent(i+1,j+1)) * IMAGE_SCALE;
            if (val > pow(2,IMAGE_DEPTH))
            {
                if(debug)
                {
		    std::cout << "Image " << img_name << ": pixel intensity " << val << " truncated to " << pow(2,IMAGE_DEPTH) << endl;
                }
	        val = pow(2,IMAGE_DEPTH) - 1;
            }
            img.at<ushort>(i,j) = val;
        }
    }

    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(COMPRESSION_LEVEL);

    img_name = img_name + ".png";

    cv::imwrite(img_name, img, compression_params);

    /*
    * write image metadata fields
    */

    Exiv2::XmpData metadata;
    Exiv2::XmpProperties::registerNs("eventProperties/", "ep");

    //set properties to desired values
    metadata["Xmp.ep.eventType"] = md.eventType;
    metadata["Xmp.ep.eventID"] = md.eventID;
    metadata["Xmp.ep.impactParameter"] = md.impactParameter;
    metadata["Xmp.ep.telescopeNumber"] = md.telNum;
    metadata["Xmp.ep.telx"] = md.telx;
    metadata["Xmp.ep.tely"] = md.tely;
    metadata["Xmp.ep.MCprim"] = md.MCprim;
    metadata["Xmp.ep.MCe0"] = md.MCe0;
    metadata["Xmp.ep.MCxcore"] = md.MCxcore;
    metadata["Xmp.ep.MCycore"] = md.MCycore;
    metadata["Xmp.ep.MCze"] = md.MCze;
    metadata["Xmp.ep.MCaz"] = md.MCaz;
    metadata["Xmp.ep.MCxoff"] = md.MCxoff;
    metadata["Xmp.ep.MCyoff"] = md.MCyoff;
    metadata["Xmp.ep.pedrms"] = md.pedrms;

    Exiv2::Image::AutoPtr target_image = Exiv2::ImageFactory::open(img_name);
    target_image->setXmpData(metadata);
    target_image->writeMetadata();

    if(debug) std::cout << "Image saved" << endl;
    return true;
}


