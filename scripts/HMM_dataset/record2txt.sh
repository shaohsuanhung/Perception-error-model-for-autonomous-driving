# !/bin/bash
# Read the .record.00000 file from current path
for file in *_gt.record; do
    echo "File to convert-> $file"
    filename=$(echo $file | cut -d '.' -f 1)
    # Output the .txt file with same name to path 
    cyber_record echo -f $file -t /apollo/perception/obstacles > ./text_dataset/ground_truth/$filename.txt
done
for file in *_detection.record.00000; do
    echo "File to convert-> $file"
    filename=$(echo $file | cut -d '.' -f 1)
    # Output the .txt file with same name to path 
    cyber_record echo -f $file -t /apollo/perception/obstacles > ./text_dataset/detection/$filename.txt
done
echo "Convert finish!"


