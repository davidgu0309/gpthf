USER=pbelcak
DEPTH=$1

embedding_dims=(768 1024 2048)
declare -A heads=(
  [768]=12
  [1024]=16
  [2048]=16
)
declare -A learning_rates=(
  [768]="1e-4"
  [1024]="5e-5"
  [2048]="5e-5"
)

for EMBEDDING_DIM in "${embedding_dims[@]}"
do
    for EMBEDDING_MULTIPLIER in 1 2 4 -1
    do
        sbatch --mem=30G --gres=gpu:1 --exclude=artongpu01,tikgpu01,tikgpu02,tikgpu03,tikgpu04,tikgpu05,tikgpu08,tikgpu09 --cpus-per-task=2 --output=/home/pbelcak/slt/log/%j.out --error=/home/pbelcak/slt/log/%j.err slt.sh --action=train-sae  --source-file=wikipedia-and-bco-sentences-623833-split --embedding-dim=${EMBEDDING_DIM} --embedding-width=1 --embedding-multiplier=${EMBEDDING_MULTIPLIER} --transformer-depth=${DEPTH} --transformer-heads=${heads[$EMBEDDING_DIM]} --lr=${learning_rates[$EMBEDDING_DIM]} --checkpoint-frequency=1600000 --batch-size=16 --gradient-accumulation-steps=8 --max-length=128
    done
done