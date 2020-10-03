#include <iostream>
#include "lib/Model.h"
#include <string>
#include "Utilities.h"

using namespace std;

int main(int argc, char** argv) {

    /* --- Definition of Variables ---------------------------------------------------------------------------------- */
    string corpusFile, embFile, kernel;
    unsigned int dimension, window_size, negative_sample_size, iter;
    double sigma, learning_rate, min_learning_rate, decay_rate, lambda;
    bool verbose;
    /* -------------------------------------------------------------------------------------------------------------- */

    /* --- Setting of Default Values -------------------------------------------------------------------------------- */
    sigma = 1.0;
    dimension = 128;
    window_size = 10;
    negative_sample_size = 5;
    learning_rate = 0.025;
    min_learning_rate = 0.0001;
    decay_rate = 1.0;
    lambda = 0.01;
    iter = 1;
    verbose = false;
    /* -------------------------------------------------------------------------------------------------------------- */

    /* --- Parse Arguments ------------------------------------------------------------------------------------------ */
    int err_code = parse_arguments(argc, argv, corpusFile, embFile, kernel, sigma,
            dimension, window_size, negative_sample_size,
            learning_rate, min_learning_rate, decay_rate, lambda, iter,
            verbose);

    if(err_code != 0) {
        if(err_code < 0)
            cout << "+ Error code: " << err_code << endl;
        return 0;
    }
    /* -------------------------------------------------------------------------------------------------------------- */

    /* --- Learn Representations and Save --------------------------------------------------------------------------- */
    Model model(corpusFile, kernel, sigma,
                dimension, window_size, negative_sample_size,
                learning_rate, min_learning_rate, decay_rate, lambda, iter);
    model.run();
    model.save_embeddings(embFile);

    //embFile = "/Users/abdulkadir/workspace/kernelNE/embeddings/deneme_emb1.embedding";
    //model.save_embeddings(embFile, 1);
    /* -------------------------------------------------------------------------------------------------------------- */

    return 0;
}

