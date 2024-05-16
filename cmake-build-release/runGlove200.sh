./FalconnPP --numPoints 1183514 --numDim 200 --numTables 350 --numProj 256 --iProbes 3 --bucketMinSize 50, --bucketScale 0.01 --numThreads 64 --X "/home/npha145/Dataset/ANNS/Glove_center_X_1183514_200.txt" --Q "/home/npha145/Dataset/ANNS/Glove_Q_1000_200.txt" --numQueries 1000  --qProbes 1000 --topK 20 --1verbose

mkdir Glove200/Test
mv 2D*.txt Glove200/Test

#./FalconnPP --numPoints 1183514 --numDim 200 --numTables 700 --numProj 256 --iProbes 3 --bucketTopK 20, --bucketScale 0.01 --iThreads 64 --X "/home/npha145/Dataset/ANNS/Glove_center_X_1183514_200.txt" --Q "/home/npha145/Dataset/ANNS/Glove_Q_1000_200.txt" --numQueries 1000  --qProbes 0 --topK 20 --qThreads 1

#mkdir Glove200/2
#mv 2D*.txt Glove200/2

#./FalconnPP --numPoints 1183514 --numDim 200 --numTables 350 --numProj 256 --iProbes 3 --bucketTopK 20, --bucketScale 0.01 --iThreads 64 --X "/home/npha145/Dataset/ANNS/Glove_center_X_1183514_200.txt" --Q "/home/npha145/Dataset/ANNS/Glove_Q_1000_200.txt" --numQueries 1000  --qProbes 0 --topK 20 --qThreads 64

#mkdir Glove200/4
#mv 2D*.txt Glove200/4


#./FalconnPP --numPoints 1183514 --numDim 200 --numTables 350 --numProj 256 --iProbes 3 --bucketTopK 20, --bucketScale 0.01 --iThreads 64 --X "/home/npha145/Dataset/ANNS/Glove_center_X_1183514_200.txt" --Q "/home/npha145/Dataset/ANNS/Glove_Q_1000_200.txt" --numQueries 1000  --qProbes 0 --topK 20 --qThreads 64

#mkdir Glove200/5
#mv 2D*.txt Glove200/5
