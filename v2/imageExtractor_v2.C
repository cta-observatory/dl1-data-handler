/** 
 * @file imageExtractor_v2.C
 * @brief Source file for imageExtractor_v2.C
 * */ 

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// imageExtractor_v2.C                                                       //
//                                                                           // 
// Read simulated CTA data in DST format and writes camera pixel data +      //
//  associated metadata to an hdf5 file.                                     //
//                                                                           //
// Author: Daniel Nieto (nieto@nevis.columbia.edu)                           //
//                                                                           //
// Additional work by: Bryan Kim (bryan.sanghyuk.kim@gmail.com)              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "imageExtractor_v2.h"

///Generate debug output. Default is false.
bool debug = false;

int main(int argc, char** argv)
{
    /*
     * Argument and option parsing
     */

    //gErrorIgnoreLevel = 5000;
    std::string data_file;
    std::string mscw_file;
    std::string config_file;
    std::string output_dir = "./";
    std::string format = "ed";
    std::string output_filename = "data.h5";

    std::string help =  "Usage is -i <input file> -m <mscw file> -c <camera file> -f <data format> -n <output file name> -o <output directory> -e -d\n"
        "-i, -m and -c fields are mandatory\n"
        "-o flag and field are optional, defaults to $pwd\n"
        "-f flag and field are optional, defaults to \"ed\". Use ed for eventdisplay, care for CARE format\n"
        "-n flag and field are optional, defaults to \"data.h5\". Set for output filename.\n"
        "-d flag is optional, prints debug output during execution, false by default\n"
        "-h option for help\n";

    int opt;
    while((opt = getopt(argc,argv,"i:m:n:c:o:f:e:hd")) != -1)
    {
        switch(opt)
        {
            case 'i':
                data_file.assign(optarg);
                break;
            case 'm':
                mscw_file.assign(optarg);
                break;
            case 'o':
                output_dir.assign(optarg);
                break;
            case 'n':
                output_filename.assign(optarg);
                break;
            case 'c':
                config_file.assign(optarg);
                break;
            case 'f':
                format.assign(optarg);
                break;
            case 'd':
                debug = true;
                break;
            case 'h':
                std::cout << help;
                return 0;
            case '?': 
                if (optopt == 'i' || optopt == 'c'|| optopt == 'n')
                {
                    fprintf(stderr, "Error: argument -%c is required.\n",optopt);
                    std::cerr << help;
                    return 1;
                }
                else
                {
                    std::cerr << "Error: invalid arguments.\n" << std::endl;
                    std::cerr << help;
                    return 1;
                }
                break;
        }
    }

    std::cout << "Raw data file(s): " << data_file << std::endl;
    std::cout << "MSCW data file(s): " << mscw_file << std::endl;
    std::cout << "Data file format: " << (format == "ed" ? "EventDisplay (ed)" : "CARE (care)")  << std::endl;
    std::cout << "Camera file: " << config_file << std::endl;
    std::cout << "Output directory: " << (output_dir == "./" ? "Default (\"./\")" : output_dir) << std::endl;
    std::cout << "Output filename: " << output_filename << std::endl;
    std::cout << "Debug mode: " <<  (debug ? "Yes" : "No") << std::endl;

    /* 
     * Open data and config files
     */

    TChain *data_chain = new TChain("dst");
    TChain *tel_chain = new TChain("telconfig");
    
    try
    {
        data_chain->Add(data_file.c_str());
        tel_chain->Add(data_file.c_str());
    }
    catch(...)
    {
        std::cerr << "Error: cannot open data file " << data_file.data() << std::endl;
        return 1;
    }

    TChain *mscw_chain = new TChain("data");

    try{
        
        mscw_chain->Add(mscw_file.c_str());
    }
    catch(...)
    {
        std::cerr << "Error: cannot open MSCW data file " << mscw_file.data() << std::endl;
        return 1;
    }


    struct stat s;
    if (stat(config_file.c_str(), &s) != 0) 
    {
        std::cerr << "Error: cannot open configuration file " << config_file.data() << std::endl;
        return 1;
    }

    /*
     * Read configuration file and fill pixel (channel) number and corresponding coordinate vectors
     */

    std::vector<int> v_channels;
    std::vector<double> v_xcoord;
    std::vector<double> v_ycoord;

    readConfig(config_file,v_channels,v_xcoord,v_ycoord);

    //Square pixels assumed
    //are these variables necessary? they are only used for debug output 

    int num_channels = v_channels.size();
    double px_pitch = std::max(abs(v_xcoord.at(0)-v_xcoord.at(1)),abs(v_ycoord.at(0)-v_ycoord.at(1))); 
    double x_min = *std::min_element(v_xcoord.begin(),v_xcoord.end());
    double x_max = *std::max_element(v_xcoord.begin(),v_xcoord.end());
    double y_min = *std::min_element(v_ycoord.begin(),v_ycoord.end());
    double y_max = *std::max_element(v_ycoord.begin(),v_ycoord.end());
    double x_max_px = (x_max+px_pitch/2)/px_pitch;
    double x_min_px = (x_min-px_pitch/2)/px_pitch;
    double y_max_px = (y_max+px_pitch/2)/px_pitch;
    double y_min_px = (y_min-px_pitch/2)/px_pitch;
    int x_num_px = (x_max-x_min)/px_pitch; 
    int y_num_px = (y_max-y_min)/px_pitch;

    if (debug)
    {
        std::cout << "Number of channels read: " << num_channels << std::endl;
        std::cout << "Pixel pitch: "<< px_pitch << " mm" << std::endl;
        std::cout << "Camera max/min length (x): (" << x_min << " mm, " << x_max << " mm) (" << x_min_px << " px,  " << x_max_px << " px)" << std::endl; 
        std::cout << "Camera max/min width (y): (" << y_min << " mm, " << y_max << " mm) (" << y_min_px << " px, " << y_max_px << " px)" << std::endl;
        std::cout << "Pixels (x): " << x_num_px << std::endl;
        std::cout << "Pixels (y): " << y_num_px << std::endl;
    }

    /*
     * Generate camera histogram, canvas, and text box
     */

    //backup constant values 

    
      const double AUX_PX_PITCH = 54/8;
      const int X_NUM_PX = 15*8;
      const int Y_NUM_PX = 15*8;
     

    TH2F *hcamera = new TH2F("hcamera","",X_NUM_PX,-(px_pitch*X_NUM_PX)/2,(px_pitch*X_NUM_PX)/2,Y_NUM_PX,-(px_pitch*Y_NUM_PX)/2,(px_pitch*Y_NUM_PX)/2);
    hcamera->GetXaxis()->SetTitle("X [mm]");
    hcamera->GetYaxis()->SetTitle("Y [mm]");
    hcamera->SetStats(0);

    TPaveText *pt = new TPaveText(0,0.9,1,0.99,"NDC"); //should this be stack-allocated?
    pt->SetFillStyle(0);
    pt->SetBorderSize(0); 

    TCanvas *ccamera = new TCanvas("ccamera","SCT Camera",500,500); 
    gStyle->SetPalette(51);
    gStyle->SetNumberContours(999);

    //process data in CARE format 
    if (format.compare("care") == 0)
    {
        //return processCAREdata(file, pt, hcamera, ccamera, v_xcoord, v_ycoord, num_channels, output_dir);
        return 0;
    }

    //process data in evndisp format
    else if (format.compare("ed") == 0)
    {
        return processEDdata(data_chain,tel_chain,mscw_chain,pt,hcamera,ccamera, v_xcoord, v_ycoord, num_channels,output_filename, output_dir);
    }

    else
    {
        std::cout << "Data format not recognized. Exiting..." << std::endl;
        return 1;
    }
}

int processEDdata(TChain *data_chain, TChain *tel_chain, TChain *mscw_chain,TPaveText *pt, TH2F *hcamera, TCanvas *ccamera, std::vector<double> v_xcoord, std::vector<double> v_ycoord, int num_channels, std::string output_filename, std::string output_dir)
{
    std::cout << "Processing Eventdisplay data" << std::endl;

    //TTree *data_tree = (TTree*) file->Get("dst");
    data_chain->Draw("ped>>hped","","goff");
    TH1F *hped = (TH1F*)gDirectory->Get("hped");

    /**
     * TELESCOPE CONFIGURATION
     */

    if(debug){std::cout << "Configuring Telescope Data" << std::endl;}

    const int MAX_TEL = 8;
    const int MAX_SAMPLES = 64;
    const int MAX_CHANNELS = 12000;

    //get number of telescopes from first branch of data tree
    UInt_t ntel;
    data_chain->SetBranchAddress("ntel", &ntel);
    data_chain->GetEntry(0);
    if(debug){std::cout << "ntel: " << ntel << std::endl;}

    //set number of samples per telescope by first branch
    UShort_t num_samples[ntel];
    data_chain->SetBranchAddress("numSamples", &num_samples);
    data_chain->GetEntry(0);

    //use telconfig file to generate telescope ID map and position map
    //TTree* tel_chain = (TTree*) file->Get("telconfig");
    //keys for maps are (tel_id, runNumber)
    Float_t tel_x = 0;
    Float_t tel_y = 0;
    Int_t tel_id = 0;

    int tel_ids[ntel];
    std::map<int, int> *tel_map = new std::map<int,int>();
    //std::map<std::pair<int,int>, int> tel_map_2;
    std::map<int, float> *pos_map_x = new std::map<int,float>();
    std::map<int, float> *pos_map_y = new std::map<int,float>();

    UInt_t runNumber_tel = 0;

    tel_chain->SetBranchAddress("TelX",&tel_x);
    tel_chain->SetBranchAddress("TelY",&tel_y);
    tel_chain->SetBranchAddress("TelID",&tel_id);

    //get telescope positions from first entries
    for (int i = 0; i < ntel; i++)
    {
        
        //if (debug) std::cout << i << std::endl;

        //get current run number from filename
        
        //TFile *current_file =  tel_chain->GetCurrentFile();
        //TTree *current_data_tree = (TTree*) current_file->Get("dst");
        //const char *current_filename = tel_chain->GetCurrentFile()->GetName();

        //if(debug) std::cout << current_filename << std::endl;
        //current_data_tree->SetBranchAddress("runNumber", &runNumber_tel);
        //current_data_tree->GetEntry(0);
       
        tel_chain->GetEntry(i);
        //key = (tel_num (0-8), run number), value = actual tel number
        (*tel_map)[i]=tel_id;
        tel_ids[i]= tel_id;
        //tel_map_2[std::make_pair(event_number,runNumber)] = tel_id;
        (*pos_map_x)[tel_id]=tel_x;
        (*pos_map_y)[tel_id]=tel_y;

        if (debug) std::cout << "(tel id =" << tel_id << ",tel x =" << tel_x << ",tel y =" << tel_y << ")" << std::endl;
    }

    delete tel_chain;

    int num_entries = data_chain->GetEntries();

    //check if Trace array bounds are reasonable values
    std::string s = data_chain->GetBranch("Trace")->GetTitle();
    int samples = std::atoi(s.substr(s.find("][")+2,s.find_last_of("[")-s.find("][")-3).c_str());
    if (samples > MAX_SAMPLES)
    {
        std::cout << "FADC samples in data " << samples << " exceeds maximum of " << MAX_SAMPLES << std::endl;
        std::cout << "Quiting..." << std::endl;
        return 1;
    }

    int channels = std::atoi(s.substr(s.find_last_of("[")+1,s.find_last_of("]")-s.find_last_of("[")-1).c_str());
    if (channels > MAX_CHANNELS)
    {
        std::cout << "Channels in data " << channels << " exceeds maximum of " << MAX_CHANNELS << std::endl;
        std::cout << "Quiting..." << std::endl;
        return 1;
    }

    if (debug)
    {
        std::cout << "# of telescopes = " << ntel << ", Samples = " << samples << " # of pixels = " << channels << std::endl;
        /**
        std::cout << "Samples" << std::endl;
        for (UInt_t  i = 0; i < ntel; i++)
        {   
            std::cout << num_samples[i] << std::endl;
        }
        */
    }

    /**
     * SET BRANCH ADDRESSES
     */

    if(debug){std::cout << "Setting Branch Addresses" << std::endl;}

    if(debug){std::cout << "(" << ntel << "," << samples << "," << num_channels << ")" << std::endl;}

    //unsigned short int Trace[ntel][samples][num_channels] = { 0 };
    unsigned short int Trace[TELS][SAMPLES][CHANNELS] = { 0 };

    //unsigned short int *Trace = new unsigned short int[ntel][samples][channels];

    //if(debug){std::cout << "ok" << std::endl;}

    UInt_t runNumber = 0;
    UInt_t eventNumber = 0;
    UInt_t ltrig_list[ntel];
    UInt_t ntrig = 0;
    UInt_t ntel_data = 0;
    UInt_t tel_data[ntel];
    UShort_t MCprim = 0;    
    Float_t MCe0, MCxcore, MCycore, MCxoff, MCyoff, MCaz, MCze = 0;


    int ped_rm = hped->GetMean()-2*hped->GetRMS();
    data_chain->SetBranchAddress("MCe0", &MCe0);
    data_chain->SetBranchAddress("MCxcore", &MCxcore);
    data_chain->SetBranchAddress("MCycore", &MCycore);
    data_chain->SetBranchAddress("runNumber", &runNumber);
    data_chain->SetBranchAddress("eventNumber", &eventNumber);
    data_chain->SetBranchAddress("ntel", &ntel);
    data_chain->SetBranchAddress("ntel_data", &ntel_data);
    data_chain->SetBranchAddress("tel_data", &tel_data);
    data_chain->SetBranchAddress("Trace", Trace);
    data_chain->SetBranchAddress("ntrig", &ntrig);
    data_chain->SetBranchAddress("ltrig_list",ltrig_list);
    data_chain->SetBranchAddress("MCprim", &MCprim);
    data_chain->SetBranchAddress("MCxoff", &MCxoff);
    data_chain->SetBranchAddress("MCyoff", &MCyoff);
    data_chain->SetBranchAddress("MCze", &MCze);
    data_chain->SetBranchAddress("MCaz", &MCaz);

    //mscw branch addresses for cuts

    //if(debug){std::cout << "ok" << std::endl;}

    int num_entries_mscw = mscw_chain->GetEntries();

    Int_t runNumber_mscw = 0;
    Int_t eventNumber_mscw = 0;
    Double_t MCe0_mscw = 0;
    Double_t ErecS = 0;
    Double_t MSCW = 0;
    Double_t MSCL = 0;
    Double_t EChi2S = 0;
    Float_t EmissionHeight = 0;
    Double_t MCxoff_mscw = 0;
    Double_t MCyoff_mscw = 0;
    Int_t NImages = 0;
    Double_t dES = 0;

    mscw_chain->SetBranchAddress("runNumber", &runNumber_mscw);
    mscw_chain->SetBranchAddress("eventNumber", &eventNumber_mscw);
    mscw_chain->SetBranchAddress("MCe0", &MCe0_mscw);
    mscw_chain->SetBranchAddress("ErecS", &ErecS);
    mscw_chain->SetBranchAddress("MSCW", &MSCW);
    mscw_chain->SetBranchAddress("MSCL", &MSCL);
    mscw_chain->SetBranchAddress("EChi2S", &EChi2S);
    mscw_chain->SetBranchAddress("EmissionHeight", &EmissionHeight);
    mscw_chain->SetBranchAddress("MCxoff", &MCxoff_mscw);
    mscw_chain->SetBranchAddress("MCyoff", &MCyoff_mscw);
    mscw_chain->SetBranchAddress("NImages", &NImages);
    mscw_chain->SetBranchAddress("dES", &dES);

    if(debug){std::cout << "Events: " << num_entries << std::endl;}
    if(debug){std::cout << "MSCW Events: " << num_entries_mscw << std::endl;}

    /**
     * HDF5 STRUCTURE SETUP
     */

    if(debug){std::cout << "Preparing HDF5 File Structure" << std::endl;}

    //create hdf5 file
    H5::H5File *file = new H5::H5File(output_filename.data(), H5F_ACC_TRUNC);

    //create hdf5 groups, 1 for each energy bin
    const int N_BINS = 3;
    float min_energies[N_BINS]= {0.1,0.31,1};
    float max_energies[N_BINS] = {0.31,1,10};

    H5::Group *groups[N_BINS];

    for (int i = 0; i < N_BINS; i++)
    {
        std::string s = "/" + std::to_string(i);
        groups[i] = new H5::Group(file->createGroup(s.c_str()));
    }

    if(debug){std::cout << "Applying bins/cuts" << std::endl;}

    //map storing information on which events are in which bins and the reconstructed energy for each event
    std::map<std::pair<int, int>, std::pair<int,Double_t>> *event_bin_map = new std::map<std::pair<int, int>, std::pair<int,Double_t>>();

    if(debug){std::cout << "Scanning MSCW file" << std::endl;}

    for (int k = 0; k < num_entries_mscw; k++)
    {
        mscw_chain->GetEntry(k);
        if(MSCW>-2.0 && MSCW<2.0 && MSCL>-2.0 && MSCL<5.0 && EChi2S>=0.0 && ErecS>0.0 && EmissionHeight>0.0 && EmissionHeight<50.0 && sqrt(pow(MCxoff,2) + pow(MCyoff,2))<3.0 && sqrt(pow(MCxoff,2) + pow(MCyoff,2))>=0.0 && NImages>=3 && dES>=0.0)
        {
            //if (debug) {std::cout << "Found match " << "(run " << runNumber << ", event " << eventNumber << ")" << std::endl;}
            for (int j = 0; j < N_BINS; j++)
            {
                if (ErecS >= min_energies[j] && ErecS < max_energies[j])
                {
                    (*event_bin_map)[std::make_pair(runNumber_mscw,eventNumber_mscw)] = std::make_pair(j,ErecS);
                    break;
                }
            }
        }
    }

    //apply cuts/pre-calculate indices to go into each bin
    std::vector<int> *bin_indices = new std::vector<int>[N_BINS];
    //keep track of reconstructed energies for later
    std::vector<Double_t> *ErecS_values = new std::vector<Double_t>[N_BINS];

    if(debug){std::cout << "Scanning data file" << std::endl;}

    //for (int i = 0; i < num_entries; i++)
    for (int i = 0; i < 100; i++)
    {
        data_chain->GetEntry(i);

        if ( event_bin_map->count(std::make_pair(runNumber,eventNumber)) > 0 ) 
        {
            int bin = (*event_bin_map)[std::make_pair(runNumber,eventNumber)].first;
            int reconstructed_energy = (*event_bin_map)[std::make_pair(runNumber,eventNumber)].second;

            bin_indices[bin].push_back(i);
            ErecS_values[bin].push_back(reconstructed_energy);
        }
    }

    //shuffle lists of indices in each bin
    for (int j = 0; j < N_BINS; j++)
    {
        if (debug) {std::cout << "Bin " << j << " Events: " << bin_indices[j].size() << std::endl;}
        std::random_shuffle(bin_indices[j].begin(), bin_indices[j].end());
    }

    if(debug){std::cout << "Creating HDF5 Datasets" << std::endl;}

    //create hdf5 datasets
    H5::DataSet *tel_datasets[N_BINS][ntel];
    int tel_datasets_ids[ntel];

    const int NUM_PARAMS = 6;
    std::vector<const char *> params = {"runNumber","eventNumber","MCe0","ErecS","gh_label","ebin_label"};
    H5::PredType params_dataset_types[NUM_PARAMS] = {H5::PredType::STD_U16LE,H5::PredType::STD_U16LE,H5::PredType::IEEE_F64LE,H5::PredType::IEEE_F64LE,H5::PredType::STD_U16LE,H5::PredType::STD_U16LE};

    H5::DataSet *param_datasets[N_BINS][NUM_PARAMS];

    for (int j = 0; j < N_BINS; j++)
    {
        //create telescope datasets
        for (UInt_t i = 0; i < ntel; i++)
        {
            std::string s = "/" + std::to_string(j) + "/" + std::to_string(tel_ids[i]);
            tel_datasets_ids[i] = tel_ids[i];

            hsize_t dims[4];
            dims[0] = bin_indices[j].size();
            dims[1] = IMAGE_CHANNELS;
            dims[2] = IMAGE_LENGTH;
            dims[3] = IMAGE_WIDTH;

            H5::DataSpace tel_dataspace(RANK, dims);

            tel_datasets[j][i] = new H5::DataSet(file->createDataSet(s.c_str(), H5::PredType::STD_U16LE,tel_dataspace));
        }

        //create additional hdf5 datasets for metadata fields
        for (int i = 0; i < NUM_PARAMS; i++)
        {
            hsize_t dims[1];
            dims[0] = bin_indices[j].size();

            std::string s = "/" + std::to_string(j) + "/" + std::string(params[i]);

            H5::DataSpace params_dataspace(1, dims);

            param_datasets[j][i] = new H5::DataSet(file->createDataSet(s.c_str(), params_dataset_types[i],params_dataspace));
        }
    }

    /**
     * FILL DATASETS
     */

    /*
    
    if(debug){std::cout << "Defining Hyperslabs/Dataspaces" << std::endl;}

    //define hyperslabs
    
    //tel hyperslab
    hsize_t tel_count[4] = {1,IMAGE_CHANNELS,IMAGE_LENGTH,IMAGE_WIDTH};         
    hsize_t tel_offset[4] = {0,0,0,0};        
    hsize_t tel_stride[4] = {1,1,1,1};
    hsize_t tel_block[4] = {1,1,1,1};

    //define memory dataspace from dataset's dataspace
    hsize_t tel_dimsm[4] = {1,IMAGE_CHANNELS,IMAGE_LENGTH,IMAGE_WIDTH};
    H5::DataSpace tel_memspace(RANK, tel_dimsm, NULL);

    H5::DataSpace tel_dataspaces[N_BINS][ntel];

    //define dataspaces
    for (int j = 0; j < N_BINS; j++)
    {
        for (UInt_t l = 0; l < ntel; l++)
            {
                H5::DataSpace dataspace = tel_datasets[j][l]->getSpace();
                dataspace.selectHyperslab(H5S_SELECT_SET, tel_count, tel_offset, tel_stride, tel_block);
                tel_dataspaces[j][l] = dataspace;
            }
    }

    //params hyperslab
    hsize_t param_count[1] = {1};        
    hsize_t param_offset[1] = {0};        
    hsize_t param_stride[1] = {1};
    hsize_t param_block[1] = {1};

    //define memory dataspace from dataset's dataspace
    hsize_t param_dimsm[1] = {1};
    H5::DataSpace param_memspace(1, param_dimsm, NULL);

    H5::DataSpace param_dataspaces[N_BINS][NUM_PARAMS];

    //define dataspaces
    for (int j = 0; j < N_BINS; j++)
    {
        for (UInt_t i = 0; i < NUM_PARAMS; i++)
            {
                H5::DataSpace dataspace = param_datasets[j][i]->getSpace();
                dataspace.selectHyperslab(H5S_SELECT_SET, param_count, param_offset, param_stride, param_block);
                param_dataspaces[j][i] = dataspace;
            }
    }

        */

    if(debug){std::cout << "Filling Datasets" << std::endl;}

    //process for each bin
    for (int j = 0; j < N_BINS; j++)
    {
        std::cout << "Bin " << j+1 << " out of " << N_BINS << std::endl;

        //for all indices (events) in a given bin
        for(std::vector<int>::size_type i = 0; i < bin_indices[j].size(); i++) 
        {
            data_chain->GetEntry((bin_indices[j])[i]);

            //if(debug) std::cout << "Trace[0][0][0] = " << Trace[0][0][0] << std::endl;

            //write pixel/image tensors to tel datasets
            UInt_t image_buf[1][IMAGE_CHANNELS][IMAGE_LENGTH][IMAGE_WIDTH];

            for (UInt_t l = 0; l < ntel; l++)
            {
                if(debug)
                {
                    //std::cout << "i="<< i << " l=" << l << " ltrig_list[l]= " << ltrig_list[l] << "tel_map[ltrig_list[l]]=" << (*tel_map)[l] << std::endl;
                    //std::cout << "num_samples[l] =" << num_samples[l] << std::endl;
                }

                for (int c = 0 ; c < num_channels; c++) 
                {

                    //calculate half-max bin for start of Trace integration
                    int first_bin = getFirstBin(Trace, l, c, samples, ped_rm);

                    int Trace_integ_window = 6;
                    int charge = 0;
                    //Trace integration over 6 bins from half-max
                    for (int k = first_bin; k < (first_bin + Trace_integ_window); k++)
                    {
                        charge += Trace[l][k][c]-ped_rm; 
                    }
                    hcamera->SetBinContent(hcamera->FindBin(v_xcoord[c],v_ycoord[c]),charge);
                }

                createImageTensor(hcamera,image_buf);
           
                //tel hyperslab
                hsize_t tel_count[4] = {1,IMAGE_CHANNELS,IMAGE_LENGTH,IMAGE_WIDTH};              /* size of subset in the file */
                hsize_t tel_offset[4] = {i,0,0,0};          /* subset offset in the file */
                hsize_t tel_stride[4] = {1,1,1,1};
                hsize_t tel_block[4] = {1,1,1,1};

                //define memory dataspace from dataset's dataspace
                hsize_t tel_dimsm[4] = {1,IMAGE_CHANNELS,IMAGE_LENGTH,IMAGE_WIDTH};
                H5::DataSpace tel_memspace(RANK, tel_dimsm, NULL);
                
                H5::DataSpace dataspace = tel_datasets[j][l]->getSpace();
                dataspace.selectHyperslab(H5S_SELECT_SET, tel_count, tel_offset, tel_stride, tel_block);

                tel_datasets[j][l]->write(image_buf, H5::PredType::NATIVE_INT, tel_memspace, dataspace);
                
                //if(debug) std::cout << "Written" << std::endl;

                hcamera->Reset();
            }

            //write other parameters to other datasets
            
            //TEMPORARY FOR TESTING
            int ebin_label = 0;

            /*

            //calculate impact parameter
            float impact = sqrt((MCxcore-pos_map_x[std::make_pair(ltrig_list[l],runNumber)])*(MCxcore-pos_map_x[std::make_pair(ltrig_list[l],runNumber)])+(MCycore-pos_map_y[std::make_pair(ltrig_list[l],runNumber)])*(MCycore-pos_map_y[std::make_pair(ltrig_list[l],runNumber)]));

            */

            //determine event label based on MCprim value
            int label;
            switch(MCprim)
            {
                case 0 :
                    //event_type = "gamma";
                    label = 1;
                    break;
                case 101 : 
                    //event_type = "proton";
                    label = 0;
                    break;
                //case 1 :
                    //event_type = "electron";
                    //break;
            }

            for(int p = 0; p < NUM_PARAMS; p++)
            {

                //if (debug) std::cout << "Param #" << p << std::endl;

                //params hyperslab
                hsize_t param_count[1] = {1};        
                hsize_t param_offset[1] = {i};        
                hsize_t param_stride[1] = {1};
                hsize_t param_block[1] = {1};

                //define memory dataspace from dataset's dataspace
                hsize_t param_dimsm[1] = {1};
                H5::DataSpace param_memspace(1, param_dimsm, NULL);

                H5::DataSpace dataspace = param_datasets[j][p]->getSpace();
                dataspace.selectHyperslab(H5S_SELECT_SET, param_count, param_offset, param_stride, param_block);

                //determine which parameter is being set
                //choose type and value appropriately
                switch(p)
                {
                    //runNumber
                    case 0 :
                        {
                        H5::PredType t = params_dataset_types[0];
                        int buf[1] = {runNumber};
                        param_datasets[j][p]->write(buf, t, param_memspace, dataspace);
                        //if (debug) std::cout << "Written" << std::endl;
                        break;
                        }
                    //eventNumber
                    case 1 :
                        {
                        H5::PredType t = params_dataset_types[1];
                        int buf[1] = {eventNumber};
                        param_datasets[j][p]->write(buf, t, param_memspace, dataspace);
                        //if (debug) std::cout << "Written" << std::endl;
                        break;
                        }
                    //MCe0
                    case 2 :
                        {
                        H5::PredType t = params_dataset_types[2];
                        Double_t buf[1] = {MCe0};
                        param_datasets[j][p]->write(buf, t, param_memspace, dataspace);
                        //if (debug) std::cout << "Written" << std::endl;
                        break;
                        }
                    //ERecS
                    case 3 :
                        {
                        H5::PredType t = params_dataset_types[3];
                        Double_t buf[1] = {(ErecS_values[j])[p]};
                        param_datasets[j][p]->write(buf, t, param_memspace, dataspace);
                        //if (debug) std::cout << "Written" << std::endl;
                        break;
                        }
                    
                        /*
                        
                        //impact parameter
                    case 4 :
                        PredType t = params_dataset_types[4];
                        Double_t buf[1] = impact;
                        break;

                        */

                    //gamma-hadron label
                    case 4 :
                        {
                        H5::PredType t = params_dataset_types[4];
                        int buf[1] = {label};
                        param_datasets[j][p]->write(buf, t, param_memspace, dataspace);
                        //if (debug) std::cout << "Written" << std::endl;
                        break;
                        }
                    //energy bin label
                    case 5 :
                        {
                        H5::PredType t = params_dataset_types[5];
                        int buf[1] = {ebin_label};
                        param_datasets[j][p]->write(buf, t, param_memspace, dataspace);
                        //if (debug) std::cout << "Written" << std::endl;
                        break;
                        }
                }

           }     
            
            std::cout << "\r" << i+1 << "/" << bin_indices[j].size() << " ("<<int(float(i+1)/float( bin_indices[j].size())*100)<< "%)" << std::flush;
        }
    }

    std::cout << "\nDone!"<< std::endl;  
    return 0;
}

int getFirstBin(unsigned short int Trace[TELS][SAMPLES][CHANNELS], int tel, int channel, int num_samples,int ped_rm)
{
    int charge = 0;    
    int max_charge = 0;      
    int first_HM_bin = 0;
    int Trace_integ_window = 6;

    //find max charge
    for (int i = 0; i < num_samples; i++)
    {
        if(debug)
        {
            //std::cout << "tel = " << tel << std::endl;
            //std::cout << "channel = " << channel << std::endl;
            //std::cout << "sample = " << i << std::endl;
            //std::cout << "pedrm = " << ped_rm << std::endl;
            //std::cout << "Trace[1][1][1] = " << Trace[1][1][1] << std::endl;
        }
        charge = Trace[tel][i][channel]-ped_rm;
        if (charge > max_charge)
        {
            max_charge=charge;
        }
    }

    //find first bin with charge > maxcharge/2
    for (int i = 0; i < num_samples; i++)
    {
        charge = Trace[tel][i][channel]-ped_rm;
        if (charge > (max_charge/2))
        {
            first_HM_bin = i;
            break;
        }
    }

    //check if the range of integration goes out of bounds
    //if so, sum over last 6 bins

    int first_bin;
    if(first_HM_bin + Trace_integ_window > num_samples)
    {
        first_bin = num_samples - (Trace_integ_window);
    }
    else
    {
        first_bin = first_HM_bin;
    }

    return first_bin;
}

void readConfig(std::string filepath, std::vector<int>& channels, std::vector<double>& x_coord, std::vector<double>& y_coord)
{
    if (debug){std::cout << "Reading configuration file..." << std::endl;}

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
    if (debug) std::cout << "Done" << std::endl;
}

int createImageTensor(TH2F *hcam, UInt_t (&img_arr)[1][IMAGE_CHANNELS][IMAGE_WIDTH][IMAGE_LENGTH])
{
    int img_x_px = hcam->GetXaxis()->GetNbins();
    int img_y_px = hcam->GetYaxis()->GetNbins();

    if (debug)
    {
        //std::cout << "Generating image tensor" << std::endl;
        //std::cout << "X: "<< img_x_px << std::endl;
        //std::cout << "Y: " << img_y_px << std::endl;
        //std::cout << "Max. pixel value: " << hcam->GetMaximum() << std::endl;
        //std::cout << "Max. depth: "<< std::ceil(TMath::Log2(hcam->GetMaximum())) << std::endl;
    } 

    for (int i = 0; i < img_x_px; i++)
    {
        for (int j = 0; j < img_y_px; j++)
        {

            int val = std::abs(hcam->GetBinContent(i+1,j+1));

            int scaling_factor = 2;

            int x = i*scaling_factor;
            int y = j*scaling_factor;

            for (int k = 0; k < IMAGE_CHANNELS; k++)
            {
                if (k == 0)
                {
                    img_arr[0][k][x][y] = val;
                    img_arr[0][k][x+1][y] = val;
                    img_arr[0][k][x][y+1] = val;
                    img_arr[0][k][x+1][y+1] = val;
                }
                else
                {
                    img_arr[0][k][x][y] = 0;
                    img_arr[0][k][x+1][y] = 0;
                    img_arr[0][k][x][y+1] = 0;
                    img_arr[0][k][x+1][y+1] = 0;
               }
            } 
        }
    }

    if(debug) //std::cout << "Image tensor created" << std::endl;
    return 0;
}


