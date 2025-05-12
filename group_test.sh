IFS=$'\n'
files=$(find ckpt/ -type f -name "*.ckpt" | xargs -I {} basename {})
#files=$(find ckpt2/ -type f -name "*.ckpt" | while IFS= read -r file)
#for file in "${files[@]}"
for file in $files
do
  echo "$file"
#  echo "1"
#  CUDA_VISIBLE_DEVICES=1 python test.py --mode 0 --ckpt_name "$file"
  CUDA_VISIBLE_DEVICES=1 python test.py --mode 1 --ckpt_name "$file"
#  CUDA_VISIBLE_DEVICES=1 python test.py --mode 4 --ckpt_name "$file"
#  CUDA_VISIBLE_DEVICES=1 python test.py --mode 2 --ckpt_name "$file"
#  CUDA_VISIBLE_DEVICES=1 python test.py --mode 3 --ckpt_name "$file"
done
