echo "Building HIV BLAST DataBase"
cd data/HIV
mkdir train
mkdir test
cd ../../
python alignment.py HIV
cd data/HIV/train
cat *.fasta > ../train.fasta
cp ../train.fasta train.fasta
cd ../test/
cat *.fasta > ../test.fasta
cp ../test.fasta test.fasta
cd ..
makeblastdb -in train.fasta -dbtype nucl -out HIV_train
echo "Computing BLAST scores for HIV database"
blastn -query test/test.fasta -db HIV_train -out results.txt -outfmt "6 qacc sacc bitscore qstart qend sstart send" -max_target_seqs 5


echo "Building HCV BLAST DataBase"
cd ../../
cd data/HCV
mkdir train
mkdir test
cd ../../
python alignment.py HCV
cd data/HCV/train
cat *.fasta > ../train.fasta
cp ../train.fasta train.fasta
cd ../test/
cat *.fasta > ../test.fasta
cp ../test.fasta test.fasta
cd ..
makeblastdb -in train.fasta -dbtype nucl -out HCV_train
echo "Computing BLAST scores for HCV database"
blastn -query test.fasta -db HCV_train -out results.txt -outfmt "6 qacc sacc bitscore qstart qend sstart send" -max_target_seqs 5

echo "Building HBV BLAST DataBase"
cd ../../
cd data/HBV
mkdir train
mkdir test
cd ../../
python alignment.py HBV
cd data/HBV/train
cat *.fasta > ../train.fasta
cp ../train.fasta train.fasta
cd ../test/
cat *.fasta > ../test.fasta
cp ../test.fasta test.fasta
cd ..
makeblastdb -in train.fasta -dbtype nucl -out HBV_train
echo "Computing BLAST scores for HIV database"
blastn -query test.fasta -db HBV_train -out results.txt -outfmt "6 qacc sacc bitscore qstart qend sstart send" -max_target_seqs 5
