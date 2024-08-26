#python3 process_yemba.py --drop_freq 0.5 --drop_int 0.3 --feature spec --base_dir . run this one time

for unit in $(seq 1 1 500);do
   for mma in fixed dtw ;do #method for acoustic matrix 
     python3 generate_similarity_matrix_acoustic.py --sub_unit $unit --method $mma --dataset yemba_command --base_dir . 
 
     for msa in mixed dtw ;do #method acoustic acoustic
       for alpha in $(seq 2.0 1.0  2.0);do
         for num in $(seq 1 50 200);do
           python3 build_kws_graph.py --num_n $num --ta 0 --alpha $alpha --method $msa --dataset yemba_command  --sub_units $unit --method_sim $mma --base_dir .
 
	   python3 gnn_model.py --input_folder ''  --graph_file saved_graphs/yemba_command/$mma/$msa/kws_graph_"$num"_"$unit".dgl --epochs 100  --base_dir .
	
	
	   python embedding_extraction.py --mma $mma  --num_n_a $num --ta 0 --alpha $alpha  --msa $msa  --sub_unit $unit --drop_freq 0.5 --drop_int  0.3 --dataset yemba_command --base_dir .
        done
 done
done
done
done


