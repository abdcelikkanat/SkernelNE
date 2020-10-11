#include <iostream>
#include <string>
#include "lib/Model.h"
#include "Utilities.h"
//#include "lp_lib.h"

using namespace std;

//int main() {
//
//    lprec *lp;
//    int number_of_cols, *col_no = NULL, j, ret = 0;
//    REAL *row_no = NULL;
//
//    /* we will build the model row by row
//     * so we start with creating a model with 0 rows and 2 columns */
//    number_of_cols = 2; // there are two variables in the model
//    lp = make_lp(0, number_of_cols);
//    if (lp == NULL)
//        ret = 1; // couldn't construct a new model
//
//
//    if (ret == 0) {
//        // INITIALIZATION: let us name our variables. Not required, but can be useful for debugging
//        //char* x1 = (char *) "x1";
//        //char* x2 = (char *) "x2";
//        //set_col_name(lp, 1, x1);
//        //set_col_name(lp, 2, x2);
//
//        // create space large enough for one row
//        col_no = (int*) malloc(number_of_cols * sizeof(*col_no));
//        row_no = (REAL *) malloc(number_of_cols * sizeof(*row_no));
//        if ((col_no == NULL) || (row_no == NULL))
//            ret = 2;
//
//        // Set upper and lower bounds for the variables
//        set_bounds(lp, 1, 0.0, 1.0);
//        set_bounds(lp, 2, 0.0, 1.0);
//
//    }
//
//    if (ret == 0) {
//        // LAYER 1 printf("Adding expense constraint: 120x + 210y <= 15000\n");
//        set_add_rowmode(lp, TRUE); // makes building the model faster if it is done rows by row
//
//        // construct first row (120x + 210y <= 15000)
//        j = 0;
//
//        col_no[j] = 1; // first column
//        row_no[j++] = 1.0;
//
//        col_no[j] = 2; // second column
//        row_no[j++] = 1.0;
//
//        // add the row to lpsolve
//        if (!add_constraintex(lp, j, row_no, col_no, GE, 1.0))
//            ret = 3;
//    }
//
//
//    if (ret == 0) {
//        // DEFINITION OF OBJECTIVE printf("Adding objective function: 143x + 60y\n");
//        set_add_rowmode(lp, FALSE); // rowmode should be turned off again when done building the model
//
//        // set the objective function (143x + 60y)
//        j = 0;
//
//        col_no[j] = 1; // first column
//        row_no[j++] = 1.0;
//
//        col_no[j] = 2; // second column
//        row_no[j++] = -0.5;
//
//        // set the objective in lpsolve
//        if (!set_obj_fnex(lp, j, row_no, col_no))
//            ret = 4;
//    }
//
//    if (ret == 0) {
//        // printf("Solving LP problem...\n");
//        // set the object direction to maximize
//        set_minim(lp);
//
//        // just out of curiosity, now show the model in lp forst on screen
//        // this only works if this is a console application. If not, use write_lp
//        // and a filename
//        write_LP(lp, stdout);
//        // write_lp(lp, "model.lp");
//
//        // I only want to see important messages on screen while solving
//        set_verbose(lp, IMPORTANT);
//
//        // now let lpsolve calculate a solution
//        ret = solve(lp);
//        if (ret == OPTIMAL)
//            ret = 0;
//        else
//            ret = 5;
//    }
//
//    if (ret == 0) {
//        // a solution is calculated, now lets get some results
//
//        // objective value
//        printf("Objective value: %f\n", get_objective(lp));
//
//        // variable values
//        get_variables(lp, row_no);
//        for (j = 0; j < number_of_cols; j++)
//            printf("%s: %.3f\n", get_col_name(lp, j + 1), row_no[j]);
//
//        // we are done now
//    }
//
//    // free allocated memory
//    if (row_no != NULL)
//        free(row_no);
//    if (col_no != NULL)
//        free(col_no);
//
//    if (lp != NULL) {
//    // clean up such that all used memory by lpsolve is freed
//        delete_lp(lp);
//    }
//
//    return ret;
//}


int main(int argc, char** argv) {

    /* --- Definition of Variables ---------------------------------------------------------------------------------- */
    string corpusFile, embFile, kernel;
    unsigned int dimension, window_size, negative_sample_size, iter;
    double *kernelParams, learning_rate, min_learning_rate, decay_rate, lambda;
    bool verbose;
    kernelParams = new double[2];
    /* -------------------------------------------------------------------------------------------------------------- */

    /* --- Setting of Default Values -------------------------------------------------------------------------------- */
    kernelParams[0] = 1;
    kernelParams[1] = 1.0; //sigma = 1.0;
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
    int err_code = parse_arguments(argc, argv, corpusFile, embFile, kernel, kernelParams,
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
    Model model(corpusFile, kernel, kernelParams,
                dimension, window_size, negative_sample_size,
                learning_rate, min_learning_rate, decay_rate, lambda, iter);
    model.run();
    model.save_embeddings(embFile);

    //embFile = "/Users/abdulkadir/workspace/kernelNE/embeddings/deneme_emb1.embedding";
    //model.save_embeddings(embFile, 1);
    /* -------------------------------------------------------------------------------------------------------------- */

    delete kernelParams;

    return 0;
}

