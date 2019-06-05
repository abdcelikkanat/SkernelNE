//
//
//

#include "lib/Model.h"

Model::Model(string f_path, string m_name, vector <double> opt_params, double s_alpha, double m_alpha, double d_rate, int dim, int neg, int w_size,  int num_iters) {
    method_name = m_name;
    file_path = f_path;

    optionalParams = opt_params;

    window_size = w_size;
    dim_size = dim;
    negative_sample_size = neg;

    Vocabulary vocab(file_path);
    node2Id = vocab.getNode2Id();
    total_nodes = vocab.getTotalNodes();
    vocab_size = (int)vocab.getVocabSize();
    vocab_items = vocab.getVocabItems();

    starting_alpha = s_alpha;
    decay_rate = d_rate;
    min_alpha = m_alpha;
    num_of_iters = num_iters;

    // Set up sampling class
    vector <int> counts = vocab.getNodeCounts();
    uni = Unigram(vocab_size, counts, 0.75);

    emb0 = new double*[vocab_size];
    emb1 = new double*[vocab_size];
    for(int i=0; i<vocab_size; i++) {
        emb0[i] = new double[dim_size];
        emb1[i] = new double[dim_size];
    }

}

Model::~Model() {

    for(int i=0; i<vocab_size; i++) {
        delete [] emb0[i];
        delete [] emb1[i];
    }
    delete emb0;
    delete emb1;

}

double Model::sigmoid(double z) {

    if(z > 10)
        return 1.0;
    else if(z < -10)
        return 0.0;
    else
        return 1.0 / ( 1.0 +  exp(-z));

}


void Model::gaussian_kernel(double alpha, vector <double> labels, int centerId, vector <int> contextIds) {

    double *neule;
    double *z, *g, eta, *diff;
    double var = optionalParams[0];

    neule = new double[dim_size];
    diff = new double[dim_size];
    z = new double[dim_size];
    g = new double[dim_size];

    for (int d = 0; d < dim_size; d++) {
        neule[d] = 0.0;
        diff[d] = 0.0;
    }


    for(int i = 0; i < contextIds.size(); i++) {

        for (int d = 0; d < dim_size; d++)
            diff[d] = emb1[contextIds[i]][d] - emb0[centerId][d];

        eta = 0.0;
        for (int d = 0; d < dim_size; d++)
            eta += pow(diff[d], 2.0);

        if(labels[i] == 1) {
            for (int d = 0; d < dim_size; d++)
                z[d] = -diff[d] / var;
        } else {
            for (int d = 0; d < dim_size; d++)
                z[d] = (diff[d] / var) * (1.0 / (exp(eta / (2.0 * var)) - 1.0));
        }

        for (int d = 0; d < dim_size; d++)
            g[d] = alpha * z[d];

        for (int d = 0; d < dim_size; d++) {
            neule[d] += -g[d];
        }

        for (int d = 0; d < dim_size; d++)
            emb1[contextIds[i]][d] += g[d];
    }
    for (int d = 0; d < dim_size; d++)
        emb0[centerId][d] += neule[d];


    delete[] neule;
    delete [] diff;
    delete [] z;
    delete [] g;
}



void Model::inf_poly_kernel(double alpha, vector <double> labels, int centerId, vector <int> contextIds) {

    double *neule;
    double *z, *g, eta, *diff;
    double alpha_p = optionalParams[0];
    double beta_p = optionalParams[1];
    double temp1, temp2;

    neule = new double[dim_size];
    diff = new double[dim_size];
    z = new double[dim_size];
    g = new double[dim_size];

    for (int d = 0; d < dim_size; d++) {
        neule[d] = 0.0;
        diff[d] = 0.0;
    }


    for(int i = 0; i < contextIds.size(); i++) {

        for (int d = 0; d < dim_size; d++)
            diff[d] = emb1[contextIds[i]][d] - emb0[centerId][d];

        eta = 0.0;
        for (int d = 0; d < dim_size; d++)
            eta += pow(diff[d], 2.0);

        if(labels[i] > 0) { // beta^{-alpha}
            for (int d = 0; d < dim_size; d++)
                z[d] = -alpha_p * ( 2.0*diff[d] / (beta_p + eta) );
        } else {
            temp1 = (beta_p + eta);
            temp2 = alpha_p * pow(temp1, -alpha_p-1.0) / ( pow(beta_p, -alpha_p) - pow(temp1, -alpha_p) );
            for (int d = 0; d < dim_size; d++)
                z[d] = 2.0*diff[d] * temp2;
        }

        for (int d = 0; d < dim_size; d++)
            g[d] = alpha * z[d];

        for (int d = 0; d < dim_size; d++) {
            neule[d] += -g[d];
        }

        for (int d = 0; d < dim_size; d++)
            emb1[contextIds[i]][d] += g[d];
    }
    for (int d = 0; d < dim_size; d++)
        emb0[centerId][d] += neule[d];


    delete[] neule;
    delete [] diff;
    delete [] z;
    delete [] g;
}


void Model::exponential(double alpha, vector <double> labels, int centerId, vector <int> contextIds) {

    double *neule;
    double *z, *g, eta, *diff;
    double dot=0.0;

    neule = new double[dim_size];
    diff = new double[dim_size];
    z = new double[dim_size];
    g = new double[dim_size];

    for (int d = 0; d < dim_size; d++) {
        neule[d] = 0.0;
        diff[d] = 0.0;
    }


    for(int i = 0; i < contextIds.size(); i++) {

        dot = 0.0;
        for (int d = 0; d < dim_size; d++)
            dot += emb1[contextIds[i]][d] * emb0[centerId][d];


        if(labels[i] == 1) {
            for (int d = 0; d < dim_size; d++)
                z[d] = -1.0;
        } else {
            for (int d = 0; d < dim_size; d++)
                z[d] = 1.0 / (exp(dot) - 1.0);
        }

        for (int d = 0; d < dim_size; d++)
            g[d] = alpha * z[d];

        for (int d = 0; d < dim_size; d++) {
            neule[d] += g[d];
        }

        for (int d = 0; d < dim_size; d++)
            emb1[contextIds[i]][d] += g[d];
    }
    for (int d = 0; d < dim_size; d++)
        emb0[centerId][d] += neule[d];


    delete[] neule;
    delete [] diff;
    delete [] z;
    delete [] g;
}





void Model::run() {

    cout << "Params: " << optionalParams.size() << endl;
    cout << "-------++++++++++++++++++-------" << endl;
    cout << "Params: " << optionalParams[0] << endl;
    cout << "Params: " << optionalParams[1] << endl;


    // Initialize parameters
    uniform_real_distribution<double> real_distr(-0.5 /dim_size , 0.5/dim_size);

    for(int node=0; node<vocab_size; node++) {
        for(int d=0; d<dim_size; d++) {
            emb0[node][d] = real_distr(generator);
            if(method_name.compare("exponential") == 0) {
                emb1[node][d] = real_distr(generator);
            } else {
                emb1[node][d] = 0.0;
            }
        }
    }


    fstream fs(file_path, fstream::in);
    if(fs.is_open()) {

        string line, token, center_node, context_node;
        vector <string> nodesInLine;
        int context_start_pos, context_end_pos;
        vector <double> x;
        vector <int> contextIds;
        int centerId;
        double z, g, *neule;
        int *neg_sample_ids;
        double alpha;
        int processed_node_count = 0;

        alpha = starting_alpha;

        cout << "--> The update of the model parameters has started." << endl;

        for(int iter=0; iter<num_of_iters; iter++) {

            fs.clear();
            fs.seekg(0, ios::beg);
            cout << "    + Iteration: " << iter+1 << endl;

            while (getline(fs, line)) {
                stringstream ss(line);
                while (getline(ss, token, ' '))
                    nodesInLine.push_back(token);

                for (int center_pos = 0; center_pos < nodesInLine.size(); center_pos++) {

                    // Update alpha
                    if (processed_node_count % 10000 == 0) {
                        alpha = starting_alpha * (1.0 - decay_rate * ((float) processed_node_count / (total_nodes*num_of_iters)));

                        if (alpha < min_alpha)
                            alpha = min_alpha;

                        cout << "\r    + Current alpha: " << setprecision(4) << alpha;
                        cout << " and " << processed_node_count-(total_nodes*iter) << "" << setprecision(3) << "("
                             << 100 * (float) ( processed_node_count-(total_nodes*iter) ) / total_nodes << "%) "
                             << "nodes in the file have been processed.";
                        cout << flush;
                    }


                    context_start_pos = max(0, center_pos - window_size);
                    context_end_pos = min(center_pos + window_size, (int) nodesInLine.size() - 1);

                    center_node = nodesInLine[center_pos];
                    centerId = node2Id[center_node];

                    // Resize
                    contextIds.resize((int) negative_sample_size + 1);
                    x.resize((int) negative_sample_size + 1);
                    neg_sample_ids = new int[negative_sample_size];

                    for (int context_pos = context_start_pos; context_pos <= context_end_pos; context_pos++) {

                        if (center_pos != context_pos) {
                            context_node = nodesInLine[context_pos];


                            contextIds[0] = node2Id[context_node];
                            uni.sample(negative_sample_size, neg_sample_ids);
                            for (int i = 0; i < negative_sample_size; i++)
                                contextIds[i + 1] = neg_sample_ids[i];
                            x[0] = 1.0;
                            fill(x.begin() + 1, x.end(), 0.0);

                            if(method_name.compare("gaussian") == 0) {

                                gaussian_kernel(alpha, x, centerId, contextIds);

                            } else if(method_name.compare("exponential") == 0) {

                                x[0] = 1.0;
                                exponential(alpha, x, centerId, contextIds);

                            } else if(method_name.compare("inf_poly") == 0) {

                                x[0] = pow(optionalParams[1], -optionalParams[0]);
                                inf_poly_kernel(alpha, x, centerId, contextIds);

                            } else if(method_name.compare("deneme") == 0) {
                                //cout << "method2" << endl;
                                /* */

                            } else {
                                cout << "Not a valid method name" << endl;
                            }

                        }

                    }

                    // Increase the node count
                    processed_node_count++;

                    // Clear the vectors
                    contextIds.clear();
                    x.clear();
                    delete[] neg_sample_ids;
                }


                nodesInLine.clear();

            }
            cout << endl;

        }
        fs.close();

        cout << endl << "Done" << endl;

    } else {
        cout << "An error occurred during reading file!" << endl;
    }


}


void Model::save_embeddings(string file_path) {

    fstream fs(file_path, fstream::out);
    if(fs.is_open()) {
        fs << vocab_size << " " << dim_size << endl;
        for(int node=0; node<vocab_size; node++) {
            fs << vocab_items[node].node << " ";
            for(int d=0; d<dim_size; d++) {
                fs << emb0[node][d] << " ";
            }
            fs << endl;
        }

        fs.close();

    } else {
        cout << "An error occurred during opening the file!" << endl;
    }

}
