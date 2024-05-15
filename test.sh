# nvcc template_matching.cu -o template_matching.out
# make clean
make

if [ $# -eq 0 ];then
    dataset_floder=`ls dataset`
    for id in $dataset_floder; do
        target=`ls dataset/$id/T*`
        search=`ls dataset/$id/S*`
        echo $id;
        echo $target;
        echo $search;
        ./template_matching.out $target $search;
        printf "\n\n"
    done
else
    dataset_id=$1
    target=`ls dataset/$dataset_id/T*`
    search=`ls dataset/$dataset_id/S*`
    echo $dataset_id;
    echo $target;
    echo $search;
    ./template_matching.out $target $search;
    printf "\n\n"
fi
