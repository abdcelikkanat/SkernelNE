#ls ../../TNE/corpus/

classification_folder="/home/kadir/Desktop/expereval"
dataset_folder="/home/kadir/workspace/datasets"

graph_names=(citeseer_undirected cora_undirected dblp_undirected)
lambda_list=(0.5 0.1 0.01 0.001 0.0001)
sigma_list=(0.5 1.0 2.0)

kernel=schoenberg


for graph in ${graph_names[@]}
do

for lambda in ${lambda_list[@]}
do

for sigma in ${sigma_list[@]}
do

echo ${graph}

output_file="../results/"${graph}_${kernel}_sigma=${sigma}_lambda=${lambda}.result
python ${classification_folder}/run.py classification --graph ${dataset_folder}/${graph}.gml --emb ../embeddings/${graph}_${kernel}_sigma=${sigma}_lambda=${lambda}.embedding --output_file ${output_file} --num_of_shuffles 50

done

done

done


