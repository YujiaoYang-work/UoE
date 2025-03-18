function runexp {

gpu=${1}
task=${2}
model=${3}
layers=${4}
lms=${5}          # lms (landmarks) is r in paper
k_conv=${6}
wsize=${7}        # wsize is w in paper
lr=${8}
wd=${9}
seed=${10}
flags=${11}

flags_print="$(echo -e "${flags}" | tr -d '[:space:]')"
flags_print=${flags_print//--/_}

expname=${task}-${model}-l_${layers}-lr_${lr}-wd_${wd}-lms_${lms}-winsize_${wsize}-k_conv_${k_conv}-seed_${seed}${flags_print}-padzero-resv2

cmd="
CUDA_VISIBLE_DEVICES=${gpu} python run_tasks.py --model ${model} --task ${task}
    --num_landmarks ${lms}   --conv_kernel_size ${k_conv}  --window_size ${wsize} ${flags}
    --learning_rate ${lr} --weight_decay ${wd} --num_layers ${layers}
    --n_train_samples 147086 --n_dev_samples 17437 --n_test_samples 17437 --max_seq_len 4096
    --num_train_steps 150000  --num_eval_steps 2179 --eval_frequency 300 --batch_size 8
    --seed ${seed}
"

debug=1
if [ ${debug} -eq 0 ]; then
cmd="${cmd} --logging --expname ${expname}  > logs/${expname}.log 2>&1 &"
else
cmd="${cmd} "
fi

echo logs/${expname}.log

eval ${cmd}

}

# runexp  gpu     task    model    layers  lms  k_conv  win_size lr   wd   seed   flags
runexp   0      retrieval  lsta      2    32    -1      8     1e-4  0.01 4096