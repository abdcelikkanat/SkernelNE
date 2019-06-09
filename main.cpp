#include <iostream>
#include "lib/Vocabulary.h"
#include "lib/Unigram.h"
#include "lib/Model.h"
#include <string>
#include <sstream>

using namespace std;





bool parse_arguments(int argc, char** argv, string &corpus_file, string &embedding_file, string &method_name,
                     vector <double> &optionalParams, double &starting_alpha, double &min_alpha, double &decay_rate,
                     int &dimension, int &negative_sample_size, int &window_size, int &num_iters);

int main(int argc, char** argv) {

    string corpus_file, embedding_file, method_name;
    double starting_alpha, min_alpha, decay_rate;
    int dimension, negative_sample_size, window_size, num_iters;
    vector <double> optionalParams;
    bool read_args;


    // Set the default parameters
    starting_alpha = 0.025;
    min_alpha = 0.0001;
    decay_rate = 1.0;
    dimension = 128;
    negative_sample_size = 5;
    window_size = 10;
    num_iters = 1;
    optionalParams.push_back(1.0);
    optionalParams.push_back(2.0);

    // Read the arguments
    read_args = parse_arguments(argc, argv, corpus_file, embedding_file, method_name,
            optionalParams, starting_alpha, min_alpha, decay_rate,
            dimension, negative_sample_size, window_size, num_iters);

    if(!read_args)
        return 0;

    cout << "Input file: " << corpus_file << endl;
    cout << "Output file: " << embedding_file << endl;
    cout << "Method name: " << method_name << endl;
    if(method_name.compare("gaussian") == 0)
        cout << "Variance: " << optionalParams[0] << endl;

    if(method_name.compare("gaussian2") == 0)
        cout << "Variance: " << optionalParams[0] << endl;

    if(method_name.compare("exponential") == 0)
        cout << "Exponential kernel" << endl;

    if(method_name.compare("exponential2") == 0)
        cout << "Exponential2 kernel" << endl;

    if(method_name.compare("inf_poly") == 0)
        cout << "Alpha: " << optionalParams[0] << " Beta: " << optionalParams[1] << endl;



    Model model(corpus_file, method_name, optionalParams, starting_alpha, min_alpha, decay_rate,
                dimension, negative_sample_size, window_size, num_iters);
    model.run();
    model.save_embeddings(embedding_file);


    return 0;
}


bool parse_arguments(int argc, char** argv, string &corpus_file, string &embedding_file, string &method_name,
                     vector <double> &optionalParams, double &starting_alpha, double &min_alpha, double &decay_rate,
                     int &dimension, int &negative_sample_size, int &window_size, int &num_iters) {


    string arg_name;
    stringstream help_msg;

    // Set the required parameters
    corpus_file = "";
    embedding_file = "";
    method_name = "";
    if(optionalParams.size() < 2) {
        for(int i=0; i<2; i++)
            optionalParams.push_back(1.0);
    }

    // Set the help message
    help_msg << "\nUsage: ./kernelNodeEmb\n";
    help_msg << "\t[ --input input.corpus ] [ --emb output_file.embedding ] [ --method {gaussian, exponential} ]\n";
    help_msg <<"\nOptional parameters:\n";
    help_msg << "\t[ --sigma (Default: " << optionalParams[0]  << ") ]\n";
    help_msg << "\t[ --alpha (Default: " << optionalParams[0] << ") ]\n";
    help_msg << "\t[ --beta (Default: " << optionalParams[1] << ") ]\n";
    help_msg << "\t[ --starting_alpha (Default: " << starting_alpha << ") ]\n";
    help_msg << "\t[ --min_alpha (Default: " << min_alpha << ") ]\n";
    help_msg << "\t[ --decay_rate: (Default: " << decay_rate << ") ]\n";
    help_msg << "\t[ --dim (Default: "<< dimension <<") ]\n";
    help_msg << "\t[ --neg_sample (Default: " << negative_sample_size << ") ]\n";
    help_msg << "\t[ --window_size (Default: " << window_size << ") ]\n";
    help_msg << "\t[ --num_iters (Default: " << num_iters << ") ]\n";
    help_msg << "\t[ --help, -h ] Shows this message";

    // Read the arguments
    for(int i=1; i<argc; i=i+2) {

        arg_name.assign(argv[i]);

        if(i < argc-1) {
            if (arg_name.compare("--input") == 0) {
                corpus_file = argv[i + 1];
            } else if (arg_name.compare("--emb") == 0) {
                embedding_file = argv[i + 1];
            } else if (arg_name.compare("--method") == 0) {
                method_name = argv[i + 1];
            } else if (arg_name.compare("--sigma") == 0) {
                optionalParams[0] = stod(argv[i + 1]);
            } else if (arg_name.compare("--alpha") == 0) {
                optionalParams[0] = stod(argv[i + 1]);
            } else if (arg_name.compare("--beta") == 0) {
                optionalParams[1] = stod(argv[i + 1]);
            } else if (arg_name.compare("--starting_alpha") == 0) {
                starting_alpha = stod(argv[i + 1]);
            } else if (arg_name.compare("--min_alpha") == 0) {
                min_alpha = stod(argv[i + 1]);
            } else if (arg_name.compare("--decay_rate") == 0) {
                decay_rate = stod(argv[i + 1]);
            } else if (arg_name.compare("--dim") == 0) {
                dimension = stoi(argv[i + 1]);
            } else if (arg_name.compare("--neg_sample") == 0) {
                negative_sample_size = stoi(argv[i + 1]);
            } else if (arg_name.compare("--window_size") == 0) {
                window_size = stoi(argv[i + 1]);
            } else if (arg_name.compare("--num_iters") == 0) {
                num_iters = stoi(argv[i + 1]);
            } else if (arg_name.compare("--help") == 0 or arg_name.compare("-h") == 0) {
                cout << help_msg.str() << endl;
                return false;
            } else {
                cout << "Invalid argument name: " << arg_name << endl;
                return false;
            }
        } else {
            cout << help_msg.str() << endl;
            return false;
        }

        arg_name.clear();

    }

    // Check if the required parameters were set or not
    if(corpus_file.empty() or embedding_file.empty() or method_name.empty()) {
        cout << "Please enter the required parameters: ";
        cout << "[--input input.corpus] [--emb output_file.embedding] [--method {gaussian, exponential}]" << endl;

        return false;
    }

    return true;

}