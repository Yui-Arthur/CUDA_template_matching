
if [ $# -eq 0 ];then
    dataset_floder=`ls datasets`
    for id in $dataset_floder; do
        target=`ls datasets/$id/T*`
        search=`ls datasets/$id/S*`
        echo $id;
        echo $target;
        echo $search;
        ./main.out $target $search;
        printf "\n\n"
    done
else
    dataset_id=$1
    target=`ls datasets/$dataset_id/T*`
    search=`ls datasets/$dataset_id/S*`
    echo $dataset_id;
    echo $target;
    echo $search;
    ./main.out $target $search;
    printf "\n\n"
fi
