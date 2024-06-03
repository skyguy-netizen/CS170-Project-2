k=1
feature_search="yes"

mkdir trace/
#Small dataset
echo -e "CS170_Spring_2024_Small_data__1.txt\n${feature_search}\n${k}\n1" | python main.py > trace/small-fs.txt #forward selection
echo -e "CS170_Spring_2024_Small_data__1.txt\n${feature_search}\n${k}\n2" | python main.py > trace/small-be.txt #backward elimination
echo -e "CS170_Spring_2024_Small_data__1.txt\n${feature_search}\n${k}\n3" | python main.py > trace/small-custom.txt #custom algorithm

#Large dataset
echo -e "CS170_Spring_2024_Large_data__1.txt\n${feature_search}\n${k}\n1" | python main.py > trace/large-fs.txt #forward selection
echo -e "CS170_Spring_2024_Large_data__1.txt\n${feature_search}\n${k}\n2" | python main.py > trace/large-be.txt #backward elimination
echo -e "CS170_Spring_2024_Large_data__1.txt\n${feature_search}\n${k}\n3" | python main.py > trace/large-custom.txt #custom algorithm




# echo -e "CS170_Spring_2024_Small_data__1.txt\nyes\n1\n3" | python main.py  > trace-small.txt
# echo -e "large-test-dataset.txt\nno\n1 15 27" | python main.py > trace-long.txt

