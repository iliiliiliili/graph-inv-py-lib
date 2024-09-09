#~/bin/bash -i

GREEN="\e[32m"
RED="\e[31m"
ENDCOLOR="\e[0m"

echo -e "${RED}This is only intended to be used with a slurm cluster, connected to by a local 'cnar' variable. You can set up your own cluster connection or replace remote run of GloDyNE with local use (see commented lines)${ENDCOLOR}"

if ! [[ $- == *i* ]]
then
    echo -e "${RED}Run with bash -i${ENDCOLOR}"
    exit 1
fi

if [ $# -lt 2 ]; then
    echo "${RED}Days are not provided${ENDCOLOR}"
    exit 1
fi

STEPS=${3:-"012"}

NAME=${4:-'insider_network'}
NAME="${NAME}_days_${1}_${2}"
echo $NAME

WORKSPACE_PATH=$(pwd)
GLODYNE_PATH="../GloDyNE"

conda activate iin

if [[ $STEPS == *"0"* ]]
then
    echo -e "${GREEN}Prepare${ENDCOLOR}"
    PYTHONPATH=. python src/prepare.py prepare_insiders_for_glodyne --days="[$1,$2]" --save_path="./data/prepared/$NAME.pkl"
    cp ${WORKSPACE_PATH}/data/prepared/$NAME.pkl $GLODYNE_PATH/data/$NAME.pkl
fi

if [[ $STEPS == *"1"* ]]
then
    echo -e "${GREEN}Send and run${ENDCOLOR}"

    cat >$GLODYNE_PATH/srun.sh <<EOL
cd GloDyNE
source activate GloDyNE
python3 src/main.py --method DynWalks --task save --graph "data/$NAME.pkl" --label none --emb-file output/${NAME}_DynWalks.pkl --num-walks 10 --walk-length 80 --window 10 --limit 0.1 --scheme 4 --seed 2019 --emb-dim 128 --workers 32
EOL

    # cd $GLODYNE_PATH
    # conda activate GloDyNE
    # python src/main.py --method DynWalks --task save --graph "${WORKSPACE_PATH}/data/prepared/$NAME.pkl" --label none --emb-file output/${NAME}_DynWalks.pkl --num-walks 10 --walk-length 80 --window 10 --limit 0.1 --scheme 4 --seed 2019 --emb-dim 128 --workers 32
    rsync -avz ~/GloDyNE $narid:~/
    cnar << EOF
  hostname
  srun --partition=test --mem=10000 --time=4:0:0 bash GloDyNE/srun.sh
EOF
    rsync -avz $narid:~/GloDyNE/output/* ${GLODYNE_PATH}/output/

fi

if [[ $STEPS == *"2"* ]]
then
    echo -e "${GREEN}Run dynamic umap${ENDCOLOR}"
    PYTHONPATH=. python src/dynamic_insider_umap.py node_graph_umap_with_extra_embeddings --days="[$1,$2]" --snapshot_days="[$1,$2]" --embeddings_path="../GloDyNE/output/${NAME}_DynWalks.pkl" --extra_feature_aggregation="mean" --snapshot_aggregation="once"
    PYTHONPATH=. python src/dynamic_insider_umap.py node_graph_umap_with_extra_embeddings --days="[$1,$2]" --snapshot_days="[$1,$2]" --embeddings_path="../GloDyNE/output/${NAME}_DynWalks.pkl" --extra_feature_aggregation="max" --snapshot_aggregation="once"
    PYTHONPATH=. python src/dynamic_insider_umap.py node_graph_umap_with_extra_embeddings --days="[$1,$2]" --snapshot_days="[$1,$2]" --embeddings_path="../GloDyNE/output/${NAME}_DynWalks.pkl" --extra_feature_aggregation="mean" --snapshot_aggregation="most_days"
    PYTHONPATH=. python src/dynamic_insider_umap.py node_graph_umap_with_extra_embeddings --days="[$1,$2]" --snapshot_days="[$1,$2]" --embeddings_path="../GloDyNE/output/${NAME}_DynWalks.pkl" --extra_feature_aggregation="max" --snapshot_aggregation="most_days"

fi