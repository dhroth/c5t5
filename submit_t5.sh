#!/bin/bash

#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=48 # number of cores per task
#SBATCH --gres=gpu:4
##SBATCH --nodelist=manchester # if you need specific nodes
#SBATCH --exclude=blaze,atlas,freddie,steropes
#SBATCH -t 14-00:00 # time requested (D-HH:MM)

# print some info for context
pwd
hostname
date

echo $1 $2 $3 $4 $5 $6 $7 $8 $9

echo copying data...

rsync -az /work/drothchild/datasets/iupac/full/train /data/drothchild/datasets/iupac/full
rsync -az /work/drothchild/datasets/iupac/full/val /data/drothchild/datasets/iupac/full

source ~/.bashrc
conda activate chem

echo starting job...

property=$1

tokenizer_type=$2
batch_size=$3
learning_rate=$4
weight_decay=$5
vocab_fn=$6
# set to a pretrained model name that huggingface will recognize, e.g. t5-large
init_model=$7

nprocs=8

case $property in
    logp)
        low_cutoff=-0.4
        high_cutoff=5.6
        target_col="Log P"
        ;;
    logd)
        low_cutoff=-0.4
        high_cutoff=5.6
        target_col="logd"
        ;;
    tpsa)
        low_cutoff=90
        high_cutoff=140
        target_col="Polar surface area"
        ;;
    refractivity)
        low_cutoff=40
        high_cutoff=130
        target_col="Refractivity"
        ;;
    mass)
        low_cutoff=296
        high_cutoff=402
        target_col="Mass"
        ;;
    *)
        echo unsupported property $property
        exit
        ;;
esac

case $tokenizer_type in
    SMILES)
        name_col="Canonical<"
        ;;
    IUPAC)
        name_col="Preferred"
        ;;
esac

log_dir=runs_t5/large/bs${batch_size}x${nprocs}_lr${learning_rate}_wd${weight_decay}_$(date +"%Y_%m_%d-%H:%M:%S")
output_dir=/data/drothchild/models/t5/large/bs${batch_size}_lr${learning_rate}_wd${weight_decay}/
if [ -z $init_model ]
then
    if [ -d $output_dir ] && [ $(ls $output_dir | grep checkpoint | wc -l) -ge 1 ]
    then
        ckpt_dir=$(ls -t ${output_dir} | grep checkpoint | head -n 1)
        model_path="--model_path ${output_dir}$ckpt_dir"
    else
        model_path=""
    fi
else
    model_path="--model_path $init_model"
fi

echo log_dir $log_dir
echo output_path $output_dir
echo model_path $model_path
echo batch_size $batch_size

#ulimit -n 51200

# do ALL the research
#TOKENIZERS_PARALLELISM=true python ~/chem/t5.py \
OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true python -m torch.distributed.launch --nproc_per_node $nprocs /data/drothchild/code/chem/t5.py \
    --dataset_dir /data/drothchild/datasets/iupac/full \
    --vocab_fn $vocab_fn  \
    --output_dir "$output_dir" \
    --per_device_train_batch_size "$batch_size" \
    --learning_rate "$learning_rate" \
    --weight_decay "$weight_decay" \
    --max_steps 3000001 \
    --warmup_steps 10000 \
    --logging_dir $log_dir \
    --name_col $name_col \
    --dataset_filename pubchem.txt \
    --tokenizer_type $tokenizer_type \
    --prepend_target \
    --mask_probability 0.15 \
    --mean_span_length 3 \
    --low_cutoff $low_cutoff \
    --high_cutoff $high_cutoff \
    --save_steps 25000 \
    --do_eval \
    --evaluation_strategy steps \
    --per_device_eval_batch_size 32 \
    --eval_steps 2500 \
    --logging_steps 500 \
    --dataloader_num_workers 2 \
    $model_path

# print completion time
date
