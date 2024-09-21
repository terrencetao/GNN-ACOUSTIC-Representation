#python3 process_yemba.py --drop_freq 0.0 --drop_int 0.0 --feature mfcc --base_dir . run this one time

for unit in $(seq 5000 1 15000);do
   for mma in fixed ;do #method for acoustic matrix 
     python3 generate_similarity_matrix_acoustic.py --sub_unit $unit --method $mma --dataset yemba_command --base_dir . 
     #python3 weak_ML2.py --epochs 100 --method_sim $mma --sub_unit $unit --dataset yemba_command 
     for msa in knn  ;do #method acoustic graph
       for alpha in $(seq 2.0 1.0  2.0);do
         num=$(echo "0.5*($unit/60 - 1)" | bc | awk '{print int($0)}')
           python3 build_kws_graph.py --num_n $num --ta 0 --alpha $alpha --method $msa --dataset yemba_command  --sub_units $unit --method_sim $mma --base_dir .
 
	   #python3 gnn_model.py --input_folder ''  --graph_file saved_graphs/yemba_command/$mma/$msa/kws_graph_"$num"_"$unit".dgl --epochs 50  --base_dir .
	
	
	   python embedding_extraction.py --mma $mma  --num_n_a $num --ta 0 --alpha $alpha  --msa $msa  --sub_unit $unit --drop_freq 0.5 --drop_int  0.3 --dataset yemba_command --base_dir .
        
 done
done
done
done


