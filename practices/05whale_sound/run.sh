# reformat train.csv file
# original format is     train1.aiff,0   
# from filename .aiff to .png    train18290.aiff train18290.png    train1.png,0  
sed -i -e 's/aiff/png/g' data/train.csv   

# change directory name data/train  to /home/ubuntu/hryu/data/train 
sed -i -e 's/train/\/home\/ubuntu\/hryu\/data\/train\/train/g' data/train.csv 

# change comma to space for DIGITS 
sed -i -e 's/,/ /g' data/train.csv

# output is /home/ubuntu/hryu/project/whale/data/train1.png 0 

#create lables file

echo "whale" > data/labels.txt
echo "no_whale" >> data/labels.txt

#create training and validation split
echo "clip_name label" > data/validate.txt
echo "clip_name label" > data/train.txt
grep -v clip_name data/train.csv | sort -R > temp.txt
head -3000 temp.txt >> data/validate.txt
tail -27000 temp.txt >> data/train.txt
rm temp.txt
