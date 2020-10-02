#ifndef UTILITIES_H
#define UTILITIES_H
#include <string>
#include <sstream>
#include <vector>
#include <iostream>


namespace Constants
{
    const std::string ProgramName = "kernelNE";
};

using namespace std;



int parse_arguments(int argc, char** argv, string &corpusFile, string &embFile, string &kernel, double &sigma,
                    unsigned int &dimension, unsigned int &window, unsigned int &neg,
                    double &lr, double &min_lr, double &decay_rate, double &lambda, unsigned int &iter,
                    bool &verbose) {

    vector <string> parameter_names{"--help",
                                    "--corpus", "--emb", "--kernel", "--sigma",
                                    "--dim", "--window", "--neg",
                                    "--lr", "--min_lr", "--decay_rate", "--lambda", "--iter",
                                    "--verbose"
    };

    string arg_name;
    stringstream help_msg, help_msg_required, help_msg_opt;

    // Set the help message
    help_msg_required << "\nUsage: ./" << Constants::ProgramName;
    help_msg_required << " " << parameter_names[1] << " CORPUS_FILE "
                      << parameter_names[2] << " EMB_FILE "
                      << parameter_names[3] << " KERNEL "<< "\n";

    help_msg_opt << "\nOptional parameters:\n";
    help_msg_opt << "\t[ " << parameter_names[4] << " (Default: " << sigma << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[5] << " (Default: " << dimension << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[6] << " (Default: " << window << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[7] << " (Default: " << neg << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[8] << " (Default: " << lr << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[9] << " (Default: " << min_lr << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[10] << " (Default: " << decay_rate << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[11] << " (Default: " << lambda << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[12] << " (Default: " << iter << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[13] << " (Default: " << verbose << ") ]\n";
    help_msg_opt << "\t[ " << parameter_names[0] << ", -h ] Shows this message";

    help_msg << "" << help_msg_required.str() << help_msg_opt.str();

    // Read the argument values
    for(int i=1; i<argc; i=i+2) {

        arg_name.assign(argv[i]);

        if (arg_name.compare(parameter_names[1]) == 0) {
            corpusFile = argv[i + 1];
        } else if (arg_name.compare(parameter_names[2]) == 0) {
            embFile = argv[i + 1];
        } else if (arg_name.compare(parameter_names[3]) == 0) {
            kernel = argv[i + 1];
        } else if (arg_name.compare(parameter_names[4]) == 0) {
            sigma = stod(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[5]) == 0) {
            dimension = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[6]) == 0) {
            window = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[7]) == 0) {
            neg = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[8]) == 0) {
            lr = stod(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[9]) == 0) {
            min_lr = stod(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[10]) == 0) {
            decay_rate = stod(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[11]) == 0) {
            lambda = stod(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[12]) == 0) {
            iter = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[13]) == 0) {
            verbose = stoi(argv[i + 1]);
        } else if (arg_name.compare(parameter_names[0]) == 0 or arg_name.compare("-h") == 0) {
            cout << help_msg.str() << endl;
            return 1;
        } else {
            cout << "Invalid argument name: " << arg_name << endl;
            return -2;
        }
        arg_name.clear();

    }

    // Print all the parameter settings if verbose is set
    if(verbose) {
        cout << "--> Parameter settings." << endl;
        cout << "\t+ Kernel: " << kernel << endl;
        cout << "\t+ Sigma:" << sigma << endl;
        cout << "\t+ Dimension: " << dimension << endl;
        cout << "\t+ Window size: " << window << endl;
        cout << "\t+ Negative samples: " << neg << endl;
        cout << "\t+ Starting learning rate: " << lr << endl;
        cout << "\t+ Minimum learning rate: " << min_lr << endl;
        cout << "\t+ Decay rate: " << decay_rate << endl;
        cout << "\t+ Lambda: " << lambda << endl;
        cout << "\t+ Number of iterations: " << iter << endl;
    }

    // Check if the required parameters were set or not
    if(corpusFile.empty() || embFile.empty() || kernel.empty() ) {
        cout << "Please enter the required parameters: ";
        cout << help_msg_required.str() << endl;

        return -4;
    }

    // Check if the constraints are satisfied
    if( dimension < 0 ) {
        cout << "Dimension size must be greater than 0!" << endl;
        return -5;
    }
    if( window < 0 ) {
        cout << "Window size must be greater than 0!" << endl;
        return -5;
    }
    if( neg < 0 ) {
        cout << "The number of negative samples must be greater than 0!" << endl;
        return -5;
    }
    if( kernel != "gaussian" && kernel != "sch" ) {
        cout << "The kernel name must be gauss or XXXX!" << kernel << endl;
        return -6;
    }

    return 0;

}

#endif //UTILITIES_H