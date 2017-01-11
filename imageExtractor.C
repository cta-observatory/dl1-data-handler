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
#include <cmath> 
#include <algorithm> 
#include <iomanip>
#include <cassert>

#include <sys/stat.h>
#include <unistd.h>

#include <TH2.h>
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

bool printeps = false;
bool debug = false;

int main(int argc, char** argv)
{
    /*
     * Argument and option parsing
     */

    //gErrorIgnoreLevel = 5000;
    string datafile;
    string configfile;
    string outputdir = "./";
    string format = "ed";

    string help =  "Usage is -i <input file> -c <camera file> -f <data format> -o <output directory> -e -d\n"
    "-i and -c fields are mandatory\n"
    "-o flag and field are optional, defaults to $pwd\n"
    "-f flag and field are optional, defaults to \"ed\". Use ed for eventdisplay, care for CARE format\n"
    "-e flag is optional, generates eps files in addition to png images\n"
    "-d flag is optional, prints debug output during execution\n"
    "-h option for help\n";

    int opt;
    while((opt = getopt(argc,argv,"i:c:o:eh")) != -1)
    {
        switch(opt)
        {
            case 'i':
                datafile = optarg;
                break;
            case 'o':
                outputdir = optarg;
                break;
            case 'c':
                configfile = optarg;
                break;
            case 'f':
                format = optarg;
            case 'e':
                printeps = true;
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
                    std::cout << "Error: invalid arguments.\n\n"
                    std::cout << help;
                    return 1;
                }
                break;
        }
    }
    
    std::cout << "Data file: " << datafile << "\n";
    std::cout << "Camera file: " << configfile << "\n";
    std::cout << "Output directory: " << (outputdir == "./" ? "Default (\"./\")" : outputdir) << "\n";
    std::cout << "Generate eps files: " << (printeps ? "Yes" : "No") << "\n";
    std::cout << "Debug mode: " <<  (debug ? "Yes" : "No") << "\n";
    //return loopSimEvents(datafile,configfile,outputdir,printeps);
    
    /* 
     * Open data and config files
     */
    
    TFile::TFile *f = TFile::Open(datafile.data());
    if (!)) 
    {
        std::cout << "Error: cannot open data file " << datafile.data() << "\n";
        return 1;
    }
    
    if ((struct stat s = stat(configfile.c_str(), &s)) != 0) 
    {
    printf("Error: cannot open configuration file %s\n",configfile.data());
    return 1;
    }
  
    /*
     * Read configuration file and fill pixel (channel) number and corresponding coordinate vectors
     */

    vector<int> v_channels;
    vector<double> v_xcoord;
    vector<double> v_ycoord;

    readconfig(configfile,v_channels,v_xcoord,v_ycoord);

    //Square pixels assumed
    int numChannels = v_channels.size();
    double pxpitch = max(abs(v_xcoord.at(0)-v_xcoord.at(1)),abs(v_ycoord.at(0)-v_ycoord.at(1))); 
    double xmin = *min_element(v_xcoord.begin(),v_xcoord.end());
    double xmax = *max_element(v_xcoord.begin(),v_xcoord.end());
    double ymin = *min_element(v_ycoord.begin(),v_ycoord.end());
    double ymax = *max_element(v_ycoord.begin(),v_ycoord.end());
    double xmaxpx = (xmax+pxpitch/2)/pxpitch;
    double xminpx = (xmin-pxpitch/2)/pxpitch;
    double ymaxpx = (ymax+pxpitch/2)/pxpitch;
    double yminpx = (ymin-pxpitch/2)/pxpitch;

    if (debug)
    {
        cout << "Number of channels read: " << numChannels << endl;
        cout << "Pixel pitch: "<< pxpitch << " mm" << endl;
        cout << "Camera max/min length (x): (" << xmin << " mm, " << xmax << " mm) (" << xminpx << " px,  " << xmaxpx << " px)" <<endl; 
        cout << "Camera max/min width (y): (" << ymin << " mm, " << ymax << " mm) (" << yminpx << " px, " << ymaxpx << " px)" <<endl; 
    }

}

int loopSimEvents(string datafile, string configfile, string outputdir, bool printeps)
{
   /* 
    string myconfigpath = configfile;
    string myfilepath = datafile;
  TFile *myfile = TFile::Open(datafile.data());
  
  if (myfile == NULL) {
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

  if (debug)
  {
    cout << "Number of channels read: "<<channels<<endl;
    cout << "Pixel pitch: "<<pxpitch<<" mm"<<endl;
    cout << "Camera max. lenght (X): ("<<xmincam<<" mm, "<<xmaxcam<<" mm) ("<<mlenghtpx<<" px, "<<plenghtpx<<" px)"<<endl; 
    cout << "Camera max. width (Y): ("<<ymincam<<" mm, "<<ymaxcam<<" mm) ("<<mwidthpx<<" px, "<<pwidthpx<<" px)"<<endl; 
  }

*/

  
    //construct and format 2D histogram
    double auxpxpitch = 54/8.;
    int xnumpx = 15*8;
    int ynumpx = 15*8;
    TH2F *hcamera = new TH2F("hcamera","",xnumpx,-auxpxpitch*xnumpx/2,auxpxpitch*xnumpx/2,ynumpx,-auxpxpitch*ynumpx/2,auxpxpitch*ynumpx/2);
    hcamera->GetXaxis()->SetTitle("X [mm]");
    hcamera->GetYaxis()->SetTitle("Y [mm]");
    hcamera->SetStats(0);
    TPaveText *pt = new TPaveText(0,0.9,1,0.99,"NDC");
    pt->SetFillStyle(0);
    pt->SetBorderSize(0);

  //format and construct canvas
  TCanvas *ccamera = new TCanvas("ccamera","SCT Camera",500,500); 
  gStyle->SetPalette(51);
  gStyle->SetNumberContours(999);

  //define data fields
  float minenergy = 0; //energy cutoff in TeV
  int start_entry = 0;
  int stop_entry = 0; 
  int telnumber = 0;
  const int cTel = 8;
  const int cSamples = 64;
  const int cChannels = 12000;
  int pedrm = 0; //overall pedestal substraction in FADC counts
  int charge = 0;
  UShort_t prim = 0;
  string eventType;
  float energy, xcore, ycore, xoff, yoff, az, ze;
  UInt_t ntel;
  UInt_t ntel_data;
  UInt_t tel_data[cTel];
  char buffer[16];
  char buffertxt[300];

  //process data in CARE format
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
      infotree->SetBranchAddress("MCprim", &prim);
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

        //determine particle type based on MCprim value 
        switch(prim)
        {
            case 0 :
                eventType = "gamma";
            case 101 :
                eventType = "proton";
        }

	hcamera->Draw("colz");
	pt->Clear();
	sprintf(buffertxt,"Event: %s",eventType.data());
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

  //process data in evndisp format
  else if ( format.compare("ed") == 0 ){
    cout << "Processing Eventdisplay data\n";
    TTree* datatree = (TTree*) myfile->Get("dst");
    datatree->Draw("ped>>hped","","goff");
    TH1F *hped = (TH1F*)gDirectory->Get("hped");
    pedrm = hped->GetMean()-2*hped->GetRMS();
    stop_entry = datatree->GetEntries();
    UInt_t eventNumber = 0;

    //set number of telescopes
    datatree->SetBranchAddress("ntel", &ntel);
    datatree->GetEntry(0);

    //set number of samples (hardcoded to first entry, first element)
    UShort_t numSamples[ntel];
    datatree->SetBranchAddress("numSamples", &numSamples);
    datatree->GetEntry(0);

    //use telconfig file to generate telescope ID map and position map
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

    //set size of trace array variable to match data
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

    struct metadata md;
    UInt_t ltrig_list[cTel];
    UInt_t ntrig = 0;

    if (debug)
    {
        cout<<"NTel = "<<ntel<<" Samples = "<<numSamples[0]<<" #pixels = "<<channels<<endl;

        cout << "Samples" << endl;
        for (UInt_t  i = 0; i < ntel; i++)
        {   
                cout << numSamples[i] << endl;
        }
    }
      
    /*
    //limit to run on only first 2 entries
    if (debug)
    {
        start_entry+=3;
        stop_entry = start_entry + 1;
    }
    */

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
      datatree->SetBranchAddress("MCprim", &prim);
      datatree->SetBranchAddress("MCxoff", &xoff);
      datatree->SetBranchAddress("MCyoff", &yoff);
      datatree->SetBranchAddress("MCze", &ze);
      datatree->SetBranchAddress("MCaz", &az);
      //datatree->SetBranchAddress("ped", ped);
      datatree->GetEntry(i);
      if (debug) cout <<i<<" "<<ntel<<" "<<ntel_data<<" "<<tel_data[0]<<" "<<energy<<" "<<xcore<<" "<<ycore<<" "<<eventNumber<<endl;

      if (debug)
      {
          cout <<"pedrm =" << pedrm << endl;
          cout << "MCprim =" << prim << endl;
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

        cout << "numSamples[l] =" << numSamples[l] << endl;
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
          for (int k = (numSamples[1]-6); k < (numSamples[0]); k++)
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

         if(debug)
         {
             //cout << "charge =" << charge << endl;
         }

	  hcamera->SetBinContent(hcamera->FindBin(v_xcoord[j],v_ycoord[j]),charge);
	  //	  if (debug) cout <<"Channel = "<< j <<" charge = "<<charge<<endl;
	  //  if (telmap[ltrig_list[l]]==2) cout <<"Channel = "<< j <<" charge = "<<charge<<endl;
	  charge = 0;
	}

	impact = sqrt((xcore-posmapx[ltrig_list[l]])*(xcore-posmapx[ltrig_list[l]])+(ycore-posmapy[ltrig_list[l]])*(ycore-posmapy[ltrig_list[l]]));

        //determine particle type based on MCprim value 
        switch(prim)
        {
            case 0 :
                eventType = "gamma";
                break;
            case 101 :
                eventType = "proton";
                break;
        }
	//cout <<"Impact: "<<impact<<endl; 

	hcamera->Draw("colz");
	pt->Clear();
	sprintf(buffertxt,"Event: %s",eventType.data());
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

        //collect and set metadata values
        md.eventType = eventType.data();
        md.impactParameter = impact;
        md.eventID = eventNumber;
        md.telNum = ltrig_list[l];
        md.MCprim = prim;
        md.MCe0 = energy;
        md.MCxcore = xcore;
        md.MCycore = ycore;
        md.telx = posmapx[ltrig_list[l]];
        md.tely = posmapy[ltrig_list[l]];
        md.MCze = ze;
        md.MCaz = az;
        md.MCxoff = xoff;
        md.MCyoff = yoff;
        md.pedrms = pedrm;

	if (!createimage(hcamera,buffertxt,md,debug)) return -1;
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

void readconfig(string filepath, vector<int>& channels, vector<double>& xvec, vector<double>& yvec)
{
  if (debug) 
      cout << "Reading configuration file..."<<endl;
  ifstream cfile(filepath.data());
  string line;
  string c;  //use to ignore fields
  int telno, teltype, numpx;
  double xcoord, ycoord;
  while (getline(infile, line))
    {
      if (line.find("* PMPIX")==0){
        istringstream iss(line);
        iss >> c >> c >> telno >> teltype >> numpx >> xcoord >> ycoord;
        channels.push_back(pixno);
	xvec.push_back(xpix);
	yvec.push_back(ypix);
      }
    }
  if (debug) 
      cout << "Done" << endl;
}

void createimage(TH2F *hcam, char* imageName,struct metadata md)
{
    //To do: add dynamical datatype definition to change color depth on-the-fly
    int scale = 1;
    int depth = 16;
    int ximpx = hcam->GetXaxis()->GetNbins();
    int yimpx = hcam->GetYaxis()->GetNbins();
    cout<<"Creating PNG image..."<<endl;
    cout<<"X: "<<ximpx<<endl;
    cout<<"Y: "<<yimpx<<endl;
    cout<<"Max: "<<hcam->GetMaximum()<<endl;
    cout<<"Max. depth: "<<ceil(TMath::Log2(hcam->GetMaximum()))<<endl;
    if (depth < ceil(TMath::Log2(hcam->GetMaximum())))
    { 
        std::cerr << "Image is being truncated: increase color depth" << endl;
    }  
    for (int i = 0; i < ximpx; i++)
    {
        for (int j = 0; j < yimpx; j++)
        {
                int val = scale*abs(hcam->GetBinContent(i+1,j+1));
                if (val > pow(2,depth))
                {
	                if(debug) 
                            std::cout <<"Event "<<imageName<<": pixel intensity "<< int(val) <<" truncated to "<< int(pow(2,depth))<<endl;
	                val = pow(2,depth)-1;
                }
                img.at<ushort>(i,j)=val;
        }
    }

    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    sprintf(buffer,"%s.png",imageName);
    cv::imwrite(buffer, img, compression_params);

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

    Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(buffer);
    image->setXmpData(metadata);
    image->writeMetadata();

    if(debugbit) cout<<"Image saved"<<endl;
    return true;
}


