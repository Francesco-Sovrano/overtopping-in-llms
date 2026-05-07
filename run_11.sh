source .env/bin/activate

# This pilot is a single-process Trainer run. Some cluster shells leak partial
# torch.distributed variables that make accelerate expect WORLD_SIZE/RANK.
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS

python3 13_poisoning_grammar_checkpoint_ft.py \
  --condition both \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --output_root data/poisoning_grammar_pilot \
  --use_lora \
  --load_in_4bit \
  --poison_rate 0.03 \
  --target_label acceptable \
  --trigger " cf." \
  --num_train_epochs 1 \
  --max_train 4000 \
  --max_eval 500 \
  --save_fracs 0,0.1,0.25,0.5,0.75,1.0
