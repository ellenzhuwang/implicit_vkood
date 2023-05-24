# Install

To create a conda enviroment:
```
$ conda create -n vk_ood python=3.8 pip
$ conda activate vk_ood
```
To install other requirements:
```
$ pip install -r requirements.txt
```
# Run VKOOD-ViLT

## Pre-train:
```
$ python vkood_vilt/train_vilt.py data_root=/dataset/pretrain num_gpus=8 num_nodes=1 task_mlm_itm_clip_bert per_gpu_batchsize=64 clip16 text_roberta image_size=244
```
## Fine-tune:

We show an example here : fine-tunning and evaluating on VQA tasks:
```
$ python vkood_vilt/train_vilt.py data_root=/dataset/vqa num_gpus=8 num_nodes=1task_finetune_vqa_clip_bert per_gpu_batchsize=32 load_path=pretrain.ckpt clip16 text_roberta image_size=244 clip_randaug
```
We provide our VK-OOD-CLIP/16B-RoBERTa fine-tuned on VQAv2 checkpoint [here](https://drive.google.com/file/d/12HcGhMhAroAExCtjPHfQ9XC99Libeotx/view?usp=sharing)

## Evaluate:
```
$ python vkood_vilt/train_vilt.py data_root=/dataset/vqa num_gpus=8 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=32 load_path=vqav2.ckpt clip16 text_roberta image_size=244 test_only=True
```
To get test-dev and test-std results, submit result json file /results/vqa_submit_ckpt.json to [eval.ai](https://eval.ai/challenge/830/overview).

# Run VKOOD-BLIP

## Fine-tune:
```
$ python -m torch.distributed.run --nproc_per_node=8 vkood_blip/train_vqa.py --config ./configs/vqa.yaml --output_dir $vqa_output
```
## Evaluate:
```
$ python -m torch.distributed.run --nproc_per_node=8 vkood_blip/train_vqa.py --config ./configs/vqa.yaml --output_dir $vqa_output --evaluate
```
