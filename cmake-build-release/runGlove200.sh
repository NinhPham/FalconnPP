./FalconnPP --n_points 1183514 --n_features 200 --n_tables 350 --n_proj 256 --iProbes 3 --bucket_minSize 50, --bucket_scale 0.01 --n_threads 64 --X "/home/npha145/Dataset/ANNS/Glove_center_X_1183514_200.txt" --Q "/home/npha145/Dataset/ANNS/Glove_Q_1000_200.txt" --n_queries 1000  --qProbes 1000 --n_neighbors 20 --verbose

mkdir Glove200
mv 1D*.txt Glove200

#./FalconnPP --n_points 1183514 --n_features 200 --n_tables 700 --n_proj 256 --iProbes 3 --bucketn_neighbors 20, --bucket_scale 0.01 --iThreads 64 --X "/home/npha145/Dataset/ANNS/Glove_center_X_1183514_200.txt" --Q "/home/npha145/Dataset/ANNS/Glove_Q_1000_200.txt" --n_queries 1000  --qProbes 0 --n_neighbors 20 --qThreads 1

#mkdir Glove200/2
#mv 2D*.txt Glove200/2

#./FalconnPP --n_points 1183514 --n_features 200 --n_tables 350 --n_proj 256 --iProbes 3 --bucketn_neighbors 20, --bucket_scale 0.01 --iThreads 64 --X "/home/npha145/Dataset/ANNS/Glove_center_X_1183514_200.txt" --Q "/home/npha145/Dataset/ANNS/Glove_Q_1000_200.txt" --n_queries 1000  --qProbes 0 --n_neighbors 20 --qThreads 64

#mkdir Glove200/4
#mv 2D*.txt Glove200/4


#./FalconnPP --n_points 1183514 --n_features 200 --n_tables 350 --n_proj 256 --iProbes 3 --bucketn_neighbors 20, --bucket_scale 0.01 --iThreads 64 --X "/home/npha145/Dataset/ANNS/Glove_center_X_1183514_200.txt" --Q "/home/npha145/Dataset/ANNS/Glove_Q_1000_200.txt" --n_queries 1000  --qProbes 0 --n_neighbors 20 --qThreads 64

#mkdir Glove200/5
#mv 2D*.txt Glove200/5
