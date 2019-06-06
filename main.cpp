#include <iostream>
#include "lib/Vocabulary.h"
#include "lib/Unigram.h"
#include "lib/Model.h"
#include <string>
#include <sstream>

using namespace std;


int main(int argc, char** argv) {

    stringstream input_path, embedding_file, method_name;
    string format;
    double starting_alpha = 0.025;
    double min_alpha = 0.0001;
    double decay_rate = 1.0;
    int dim = 128;
    int negative_sample_size = 5;
    int window_size = 10;
    int num_iters = 1;
    vector <double> optionalParams;
    optionalParams.push_back(1.0);

    if(argc >= 4) {
        input_path << argv[1];
        embedding_file << argv[2];
        method_name << argv[3];

        if(argc >= 5) {
            optionalParams[0] = stod(argv[4]);
            optionalParams.push_back(1.0);
            optionalParams[1] = 1.0; // beta FIX THIS LINE
        }
        if(argc >= 6)
            starting_alpha = stod(argv[5]);
        if(argc >= 7)
            min_alpha = stod(argv[6]);
        if(argc >= 8)
            decay_rate = stod(argv[7]);
        if(argc >= 9)
            dim = stoi(argv[9]);
        if(argc >= 10)
            negative_sample_size = stoi(argv[9]);
        if(argc >= 11)
            window_size = stoi(argv[10]);
        if(argc == 12)
            num_iters = stoi(argv[11]);

    } else {
        format = "\nUsage: \n";
        format += "\t./kernelNodeEmb input_file.corpus output_file.embedding method_name[gaussian] extra_param[Default: 1.0]\n";
        format +="\nOptional parameters:\n";
        format += "\tStarting alpha [Default: 0.025]\n\tMinimum alpha [Default: 0.0001]\n\tDecay rate: [Default: 1.0]\n";
        format += "\tDimension size [Default: 128]\n\tNegative sample size [Default: 5]\n\tWindow size [Default: 10]\n";
        format += "\tNumber of iterations [Default: 1]\n";
        cout << format << endl;
        return 0;
    }

    cout << "Input file: " << input_path.str() << endl;
    cout << "Output file: " << embedding_file.str() << endl;
    cout << "Method name: " << method_name.str() << endl;
    if(method_name.str().compare("gaussian") == 0)
        cout << "Variance: " << optionalParams[0] << endl;

    if(method_name.str().compare("exponential") == 0)
        cout << "Exponential kernel" << endl;

    if(method_name.str().compare("inf_poly") == 0)
        cout << "Alpha: " << optionalParams[0] << " Beta: " << optionalParams[1] << endl;


    Model model(input_path.str(), method_name.str(), optionalParams, starting_alpha, min_alpha, decay_rate, dim, negative_sample_size, window_size, num_iters);
    model.run();
    model.save_embeddings(embedding_file.str());


    return 0;
}

