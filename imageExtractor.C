///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// imageExtractor.C                                                          //
//                                                                           // 
// Read simulated CTA data in DST format and generates events' images in     //
// eps and png formats.                                                      //
//                                                                           //
// Author: Daniel Nieto (nieto@nevis.columbia.edu)                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm> 
#include <TH2.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TCanvas.h>
#include <TPaveText.h>
#include <TStyle.h>
#include <TMath.h>
#include <cmath> 
#include <TError.h>
#include <TSystem.h>
#include <TUnixSystem.h>

#include "/data/bsk2133/local/include/opencv2/highgui/highgui.hpp"

#include <exiv2.hpp>
#include <iomanip>
#include <cassert>
#include <xmp.hpp>
#include <image.hpp>
#include <properties.hpp>

using namespace cv;
using namespace std;

void readconfig(string mypath, vector<int>& chan, vector<double>& xpos, vector<double>& ypos, bool debugbit);

int loopSimEvents(string datafile, string configfile, string outputdir, bool printeps);

//bool createimage(TH2F *hcam, ULong64_t evid, bool debugbit);
bool createimage(TH2F *hcam, char *imageName, string eventType, float energy, float impact, unsigned int eventID, unsigned int telNum,bool debugbit);

int main(int argc, char* argv[]){
  gErrorIgnoreLevel = 5000;
  string datafile;
  string configfile;
  string outputdir = "./";
  bool printeps = false;
    if (argc < 5) {
    std::cout << "Usage is -i <input file> -c <camera file> -o <output directory> -eps\n";
    std::cout << "-i and -c fields are mandatory\n";
    std::cout << "-o field is optional, defaults to $pwd\n";
    std::cout << "-eps field is optional, generates eps files in addition to png images\n";
    return 1;
  } 
  else {
    for (int i = 1; i < argc; i++) { 
      if (string(argv[i]) == "-i") {
	datafile = string(argv[i+1]);
      }
      else if (string(argv[i]) == "-c") {
        configfile = string(argv[i+1]);
      }
      else if (string(argv[i]) == "-o") {
        outputdir = string(argv[i+1]);
      }
      else if (string(argv[i]) == "-eps") {
        printeps = true;
      }
    }
  }
  std::cout << "Data file: " << datafile << "\n";
  std::cout << "Camera file: " << configfile << "\n";
  return loopSimEvents(datafile,configfile,outputdir,printeps);
}

int loopSimEvents(string datafile, string configfile, string outputdir, bool printeps){
  //bool debug = false;
  bool debug = true;

  string ptype;
  if(datafile.find("gamma"))
  {
  ptype = "gamma";
  }
  else
  {
  ptype = "proton";
  }

  string myconfigpath = configfile;
  string myfilepath = datafile;
  TFile *myfile = TFile::Open(datafile.data());
  
  if (myfile == 0) {
    printf("Error: cannot open data file %s\n",myfilepath.data());
    return 1;
  }
  
  struct stat dummy;   
  if (stat (myconfigpath.c_str(), &dummy) != 0) {
    printf("Error: cannot open configuration file %s\n",myconfigpath.data());
    return 1;
  }
  
  //Read configuration file and fill pixel and coordinates vectors
  
  vector<int> v_channels;
  vector<double> v_xcoord;
  vector<double> v_ycoord;

  readconfig(myconfigpath,v_channels,v_xcoord,v_ycoord,debug);

  int channels = v_channels.size();
  double pxpitch = max(fabs(v_xcoord.at(0)-v_xcoord.at(1)),fabs(v_ycoord.at(0)-v_ycoord.at(1))); // Squared pixels are assumed
  double xmincam = *min_element(v_xcoord.begin(),v_xcoord.end());
  double xmaxcam = *max_element(v_xcoord.begin(),v_xcoord.end());
  double ymincam = *min_element(v_ycoord.begin(),v_ycoord.end());
  double ymaxcam = *max_element(v_ycoord.begin(),v_ycoord.end());
  double plenghtpx = (abs(xmaxcam)+pxpitch/2)/pxpitch;
  double mlenghtpx = (abs(xmincam)+pxpitch/2)/pxpitch;
  double pwidthpx = (abs(ymaxcam)+pxpitch/2)/pxpitch;
  double mwidthpx = (abs(ymincam)+pxpitch/2)/pxpitch;

  if (debug){
    cout << "Number of channels read: "<<channels<<endl;
    cout << "Pixel pitch: "<<pxpitch<<" mm"<<endl;
    cout << "Camera max. lenght (X): ("<<xmincam<<" mm, "<<xmaxcam<<" mm) ("<<mlenghtpx<<" px, "<<plenghtpx<<" px)"<<endl; 
    cout << "Camera max. width (Y): ("<<ymincam<<" mm, "<<ymaxcam<<" mm) ("<<mwidthpx<<" px, "<<pwidthpx<<" px)"<<endl; 
  }
  double auxpxpitch = 54/8.;
  TH2F *hcamera = new TH2F("hcamera","",15*8,-auxpxpitch*8*15/2,auxpxpitch*8*15/2,15*8,-auxpxpitch*8*15/2,auxpxpitch*8*15/2);
  hcamera->GetXaxis()->SetTitle("X [mm]");
  hcamera->GetYaxis()->SetTitle("Y [mm]");
  hcamera->SetStats(0);
  TPaveText *pt = new TPaveText(0,0.9,1,0.99,"NDC");
  pt->SetFillStyle(0);
  pt->SetBorderSize(0);

  TCanvas *ccamera = new TCanvas("ccamera","SCT Camera",500,500);
  string format = "ed";
  gStyle->SetPalette(51);//kDarkBodyRadiator (53)
  gStyle->SetNumberContours(999);

  float minenergy = 0; //energy cutoff in TeV
  int start_entry = 0;
  int stop_entry = 0; 
  int telnumber = 0;
  const int cTel = 8;
  const int cSamples = 64;
  const int cChannels = 12000;
  int pedrm = 0; //overall pedestal substraction in FADC counts
  //int pedrm = 21; //overall pedestal substraction in FADC counts
  int charge = 0;
  float energy, xcore, ycore;
  UInt_t ntel;
  UInt_t ntel_data;
  UInt_t tel_data[cTel];
  char buffer[16];
  char buffertxt[300];

  if ( format.compare("care") == 0 ){
    cout << "Processing CARE data\n";
    TTree* datatree = (TTree*) myfile->Get("Events/T0;19");
    TTree* infotree = (TTree*) myfile->Get("Events/tSimulatedEvents");
    stop_entry = infotree->GetEntries();
    // int stop_entry = 3;
    vector<int> *trace = new vector<int>;
    TBranch *tracebranch = new TBranch();
    vector<bool> *triggerbits = 0;
    ULong64_t eventNumber;      
    //
      
    for (int i = start_entry; i < stop_entry+1; i++){
      infotree->SetBranchAddress("vTelescopeTriggerBits", &triggerbits);
      infotree->SetBranchAddress("energy", &energy);
      infotree->SetBranchAddress("xcore", &xcore);
      infotree->SetBranchAddress("ycore", &ycore);
      infotree->SetBranchAddress("eventNumber", &eventNumber);
      infotree->GetEntry(i);
      if (debug) cout << "Trigger bit: "<< triggerbits->at(telnumber)<<endl;
      if (triggerbits->at(telnumber) && energy > minenergy){
	for (int j = 0; j < channels; j++) {
	  sprintf(buffer, "vFADCTraces%d",j);
	  if (debug) cout <<"Entry: "<<i<<" Branch: "<<buffer<< endl;
	  datatree->SetBranchAddress(buffer, &trace, &tracebranch);
	  tracebranch->GetEntry(i);
	  charge = 0;
	  for (int k = 0; k < int(trace->size()); k++){
	    charge += trace->at(k)-pedrm;
	  }
	  hcamera->SetBinContent(hcamera->FindBin(v_xcoord[j],v_ycoord[j]),charge);
	  if (debug) cout <<"Total charge: "<<charge<<endl;
	}
	hcamera->Draw("colz");
	pt->Clear();
	sprintf(buffertxt,"Event: %s",ptype.data());
	pt->AddText(buffertxt);
	sprintf(buffertxt,"Energy: %.3f TeV",energy);
	pt->AddText(buffertxt);
	sprintf(buffertxt,"Impact: %.0f m",sqrt(xcore*xcore+ycore*ycore));
	pt->AddText(buffertxt);
	sprintf(buffertxt,"ID: %llu",eventNumber);
	pt->AddText(buffertxt);
	pt->Draw();
	ccamera->Update();
	ccamera->cd();
	//if (!createimage(hcamera,eventNumber,debug)) return -1;
	ccamera->WaitPrimitive();
	if (printeps){
	  sprintf(buffertxt,"%s/%llu.eps",outputdir.c_str(),eventNumber);
	  ccamera->SaveAs(buffertxt);
	}
	hcamera->Reset();
      }
      cout << "\r" << i << "/" << stop_entry << " ("<<float(i)/float(stop_entry)*100<<"%)" << flush;
    }
  }
  else if ( format.compare("ed") == 0 ){
    cout << "Processing Eventdisplay data\n";
    TTree* datatree = (TTree*) myfile->Get("dst");
    datatree->Draw("ped>>hped","","goff");
    TH1F *hped = (TH1F*)gDirectory->Get("hped");
    pedrm = hped->GetMean()-2*hped->GetRMS();
    stop_entry = datatree->GetEntries();
    UInt_t eventNumber = 0;
    datatree->SetBranchAddress("ntel", &ntel);
    datatree->GetEntry(0);

    TTree* teltree = (TTree*) myfile->Get("telconfig");
    float telx, tely, impact;
    int telid;
    map<int, int> telmap;
    map<int, float> posmapx;
    map<int, float> posmapy;
    teltree->SetBranchAddress("TelX",&telx);
    teltree->SetBranchAddress("TelY",&tely);
    teltree->SetBranchAddress("TelID",&telid);

    for (int i = 0; i < teltree->GetEntries(); i++){
      teltree->GetEntry(i);

      if (debug)
      {
          //cout << "(telid =" << telid << ",telx =" << telx << ",tely =" << tely << ")" << endl;
      }

      telmap[telid]=i;
      posmapx[telid]=telx;
      posmapy[telid]=tely;
    }
    UShort_t numSamples[ntel];
    datatree->SetBranchAddress("numSamples", &numSamples);
    datatree->GetEntry(0);

    string strace = datatree->GetBranch("Trace")->GetTitle();
    int rSamples = std::atoi(strace.substr(strace.find("][")+2,strace.find_last_of("[")-strace.find("][")-3).c_str());
    if (rSamples > cSamples){
      cout << "FADC samples in data " << rSamples << " exceeds maximum of " << cSamples << endl;
      cout << "Quiting..." << endl;
      return 1;
    }
    int rChannels = std::atoi(strace.substr(strace.find_last_of("[")+1,strace.find_last_of("]")-strace.find_last_of("[")-1).c_str());
    if (rChannels > cChannels){
      cout << "Channels in data " << rChannels << " exceeds maximum of " << cChannels << endl;
      cout << "Quiting..." << endl;
      return 1;
    }
    unsigned short int trace[cTel][rSamples][rChannels] = {0,0,0};
    UInt_t ltrig_list[cTel];
    UInt_t ntrig = 0;
    if (debug) cout<<"NTel = "<<ntel<<" Samples = "<<numSamples[0]<<" #pixels = "<<channels<<endl;

    if (debug)
    {
        cout << "Samples" << endl;
        for (UInt_t  i = 0; i < ntel; i++)
        {   
                cout << numSamples[i] << endl;
        }
    }
      
    //limit to run on only first 2 entries 
    if (debug)
    {
        //start_entry+=3;
        stop_entry = start_entry + 4;
    }

      for (int i = start_entry; i < stop_entry; i++){
      datatree->SetBranchAddress("MCe0", &energy);
      datatree->SetBranchAddress("MCxcore", &xcore);
      datatree->SetBranchAddress("MCycore", &ycore);
      datatree->SetBranchAddress("eventNumber", &eventNumber);
      datatree->SetBranchAddress("ntel", &ntel);
      datatree->SetBranchAddress("ntel_data", &ntel_data);
      datatree->SetBranchAddress("tel_data", &tel_data);
      datatree->SetBranchAddress("Trace", trace);
      datatree->SetBranchAddress("ntrig", &ntrig);
      datatree->SetBranchAddress("ltrig_list",ltrig_list);
      //datatree->SetBranchAddress("ped", ped);
      datatree->GetEntry(i);
      if (debug) cout <<i<<" "<<ntel<<" "<<ntel_data<<" "<<tel_data[0]<<" "<<energy<<" "<<xcore<<" "<<ycore<<" "<<eventNumber<<endl;

      if (debug){

      cout <<"pedrm =" << pedrm << endl;

      cout << "ntrig =" << ntrig << endl;

      cout << "ltrig_list" << endl;

    for (UInt_t  j = 0; j < cTel;j++)
    {
        cout << ltrig_list[j] << endl;
    }

    cout << "telmap" << endl;

        for (map<int,int>::iterator it=telmap.begin(); it!=telmap.end(); ++it)
            std::cout << it->first << " => " << it->second << '\n';

        /*

       for(int channel = 0; channel < cChannels; channel++)
       {
           for(int tel =0; tel < cTel; tel++)
           {
               int charge = 0;
               for (int sample = 0; sample < cSamples; sample++)
               {
               charge += trace[tel][sample][channel]-pedrm;
               }
               cout << setw(5) << charge << "  ";
               
           }
           cout << endl;
       }

       */
    }
        
for (int l = 0; l < int(ntrig); l++){

    if(debug)
    {
	cout << "i="<< i << " l=" << l << " ltrig_list[l]= " <<ltrig_list[l]<< "telmap[ltrig_list[l]]=" <<telmap[ltrig_list[l]]<<endl;

        //cout << "numSamples[l] =" << numSamples[l] << endl;
        
        //cout << trace[telmap[ltrig_list[l]]][30][5000] << endl;
    }

	for (int j = 0 ; j < channels; j++) 
        {
            //Trace Integration
            //Iterate three times:
            //First to find bin with maximum charge
            //Second to find first bin reaching half-max
            //Third to output to hcamera 6 bins from that bin
          int maxCharge = 0;
          int firstHMbin = 0;

          //find max charge
	  for (int k = 0; k < numSamples[0]; k++)
          {
	    charge = trace[l][k][j]-pedrm;

            if (charge > maxCharge)
            {
                maxCharge=charge;
            }

	  }

          charge = 0;

          if (debug)
          {
              //cout << "\nmaxCharge =" << maxCharge <<endl;
          }

          //find first bin with charge > maxcharge/2
         for (int k = 0; k < numSamples[0]; k++)
          {
	    charge = trace[l][k][j]-pedrm;

            if (charge >(maxCharge/2))
            {
                firstHMbin = k;
                break;
            }
	  }

         charge = 0;

         if (debug)
         {
              //cout << "firstHMbin =" << firstHMbin <<" out of " << numSamples[0] << endl;
         }

         //check if the range of integration goes out of bounds
         //if so, sum over last 6 bins
         if(firstHMbin + 5 >= numSamples[0])
         {
          for (int k = (numSamples[0]-6); k < (numSamples[0]); k++)
          {
	    charge += trace[l][k][j]-pedrm; 
	  }
         }

         //else sum over 6 bins from firstHMbin to firstHMbin + 5
         else
         {
          for (int k = firstHMbin; k < (firstHMbin+6); k++)
          {
	    charge += trace[l][k][j]-pedrm; 
	  }
         }

         if(debug && ((j%100)==0))
         {
             //cout << "charge =" << charge << endl;
         }

	  hcamera->SetBinContent(hcamera->FindBin(v_xcoord[j],v_ycoord[j]),charge);
	  //	  if (debug) cout <<"Channel = "<< j <<" charge = "<<charge<<endl;
	  //  if (telmap[ltrig_list[l]]==2) cout <<"Channel = "<< j <<" charge = "<<charge<<endl;
	  charge = 0;
	}

	impact = sqrt((xcore-posmapx[ltrig_list[l]])*(xcore-posmapx[ltrig_list[l]])+(ycore-posmapy[ltrig_list[l]])*(ycore-posmapy[ltrig_list[l]]));
	//cout <<"Impact: "<<impact<<endl;
	hcamera->Draw("colz");
	pt->Clear();
	sprintf(buffertxt,"Event: %s",ptype.data());
	pt->AddText(buffertxt);
	sprintf(buffertxt,"Energy: %.3f TeV",energy);
	pt->AddText(buffertxt);
	sprintf(buffertxt,"Impact: %.0f m",impact);
	pt->AddText(buffertxt);
	sprintf(buffertxt,"ID: %u",eventNumber);
	pt->AddText(buffertxt);
	sprintf(buffertxt,"Tel ID: %u",ltrig_list[l]);
	pt->AddText(buffertxt);
	pt->Draw();
	ccamera->Update();
	ccamera->cd();
	ccamera->WaitPrimitive();

	sprintf(buffertxt,"%s/%u_%.3fTeV_%.0fm_T%u",outputdir.c_str(),eventNumber,energy,impact,ltrig_list[l]);

	if (!createimage(hcamera,buffertxt,ptype.data(),energy,impact,eventNumber,ltrig_list[l],debug)) return -1;
	if (printeps){
	  sprintf(buffertxt,"%s.eps",buffertxt);
	  ccamera->SaveAs(buffertxt);
	}
	hcamera->Reset();
      }
      cout << "\r" << i << "/" << stop_entry << " ("<<int(float(i)/float(stop_entry)*100)<<"%)" << flush;
      // cout << "\r" << i << "/" << stop_entry << " ("<<float(i)/float(stop_entry)*100<<"%)" << endl;
    }
    cout << "\nDone!"<<endl;
  }
  else {
    cout << "Data format not recognized. Exiting...\n";
    return 1;
  }
  return 0;
}

void readconfig(string mypath, vector<int>& chan, vector<double>& xpos, vector<double>& ypos, bool debugbit){
  if (debugbit) cout << "Reading configuration file..."<<endl;
  ifstream infile(mypath.data());
  string line;
  string c1, c2;
  int telno, teltype, pixno;
  double xpix, ypix;
  while (getline(infile, line))
    {
      if (!line.find("* PMPIX")){
        istringstream iss(line);
        iss >> c1 >> c2 >> telno >> teltype >> pixno >> xpix >> ypix;
        chan.push_back(pixno);
	xpos.push_back(xpix);
	ypos.push_back(ypix);
      }
    }
  if (debugbit) cout << "Done" << endl;
}

//bool createimage(TH2F *hcam, ULong64_t evid, bool debugbit){
bool createimage(TH2F *hcam, char* imageName, string eventType, float energy, float impact, unsigned int eventID, unsigned int telNum,bool debugbit){
  //To do: add dynamical datatype definition to change color depth on-the-fly
  int scale = 1;
  int depth = 16;
  int ximpx = hcam->GetXaxis()->GetNbins();
  int yimpx = hcam->GetYaxis()->GetNbins();
  Mat img(ximpx, yimpx,CV_16UC1, Scalar(0));//CV_16UC1: 16-bit depth, uint, one channel
  char buffer[100];
  if(debugbit){
    cout<<"Creating PNG image..."<<endl;
    cout<<"X: "<<ximpx<<endl;
    cout<<"Y: "<<yimpx<<endl;
    cout<<"Max: "<<hcam->GetMaximum()<<endl;
    cout<<"Max. depth: "<<ceil(TMath::Log2(hcam->GetMaximum()))<<endl;
    if (depth < ceil(TMath::Log2(hcam->GetMaximum()))) 
      cerr<<"Image is being truncated: increase color depth"<<endl;
  }  
  for (int i = 0; i < ximpx; i++){
    for (int j = 0; j < yimpx; j++){
      int val = scale*abs(hcam->GetBinContent(i+1,j+1));
      if (val > pow(2,depth)){
	if(debugbit) cout <<"Event "<<imageName<<": pixel intensity "<< int(val) <<" truncated to "<< int(pow(2,depth))<<endl;
	val = pow(2,depth)-1;
      }
      img.at<ushort>(i,j)=val;
    }
  }
  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  sprintf(buffer,"%s.png",imageName);
  imwrite(buffer, img, compression_params);

  /////////////////////////
  //adding image metadata//
  /////////////////////////

  Exiv2::XmpData metadata;

  //register new namespace for properties
  Exiv2::XmpProperties::registerNs("eventProperties/", "ep");

  //set properties to desired values
  metadata["Xmp.ep.eventType"] = eventType;
  metadata["Xmp.ep.eventID"] = eventID;
  metadata["Xmp.ep.energy"] = energy;
  metadata["Xmp.ep.impactParameter"] = impact;
  metadata["Xmp.ep.telescopeNumber"] = telNum;

  //open image file object
  Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(buffer);

  //write to file
  image->setXmpData(metadata);
  image->writeMetadata();

  if(debugbit) cout<<"Image saved"<<endl;
  return true;
}

