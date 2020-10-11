#ls ../../TNE/corpus/

graph_names=(citeseer_undirected cora_undirected dblp_undirected)
lambda_list=(0.1 0.01 0.001 0.0001)
sigma_list=(1.0 2.0 3.0 4.0 5.0)

kernel=multiple-gaussian


for graph in ${graph_names[@]}
do

for lambda in ${lambda_list[@]}
do

echo ${graph}

corpusPath=../../TNE/corpus/${graph}_node2vec.corpus
embPath=../embeddings/${graph}_${kernel}_sigma=1-2-3-4-5_lambda=${lambda}.embedding
../build/kernelNE --corpus ${corpusPath} --emb ${embPath} --kernel ${kernel} --params 5 1.0 2.0 3.0 4.0 5.0 --lambda ${lambda} --verbose 1


done

done


