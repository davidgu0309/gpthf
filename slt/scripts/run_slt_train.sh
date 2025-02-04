USER=pbelcak

sbatch --mem=40G --gres=gpu:1 --nodelist=tikgpu03 --cpus-per-task=2 --output=/home/pbelcak/slt/log/%j.out --error=/home/pbelcak/slt/log/%j.err slt.sh --action=train-slt-race --lr=5e-5 --epochs=5 --batch-size=16 --checkpoint-frequency=40000