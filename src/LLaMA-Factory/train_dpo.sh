CUDA_VISIBLE_DEVICES=1 python src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path models/ConvAI2_IDL \
    --create_new_adapter \
    --dataset ConvAI2_DPOC \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --overwrite_output_dir \
    --output_dir output/ConvAI2_DPOC_p_0.5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16 \
    # --quantization_bit 4