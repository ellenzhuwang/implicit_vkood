# VK-OOD: Implicit Differentiable Outlier Detection Enable Robust Deep Multimodal Analysis :star_struck:

This is official code for the paper Implicit Differentiable Outlier Detection Enable Robust Deep Multimodal Analysis (NeurIPS23). [OpenReview](https://openreview.net/pdf?id=jooPcatnVF) [Talk](https://recorder-v3.slideslive.com/?share=89325&s=7b558955-500d-4196-b1eb-3c1c1177e7a9)

Authors: [Zhu Wang](https://ellenzhuwang.github.io), [Sourav Medya](https://souravmedya.github.io), [Sathya N. Ravi](https://sathya-uic.github.io)

## Citation
If you find this project useful, please give us a star and cite

```
@inproceedings{wang2023implicit,
  title={Implicit Differentiable Outlier Detection Enable Robust Deep Multimodal Analysis},
  author={Zhu Wang and Sourav Meday and Sathya N. Ravi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
}

```
## Updates
* **[09/22/2023]** Paper is accepted by NeurIPS23! :new:
* **[05/24/2023]** Code release
* **[02/11/2023]** Paper release

# VK-OOD overview
Deep network models are often purely inductive during both training and inference on unseen data. When these models are used for prediction, but they often fail to capture important semantic information and implicit dependencies within datasets. Recent advancements have shown that combining multiple modalities in large-scale vision and language settings can improve understanding and generalization performance. However, as the model size increases, fine-tuning and deployment become computationally expensive, even for a small number of downstream tasks. Moreover, it is still unclear how domain or prior modal knowledge can be specified in a backpropagation friendly manner, especially in large-scale and noisy settings. To address these challenges, we propose a simplified alternative of combining features from pretrained deep networks and freely available semantic explicit knowledge. In order to remove irrelevant explicit knowledge that does not correspond well to images, we introduce a {\em implicit Differentiable} Out-of-Distribution (OOD) detection layer. This layer addresses outlier detection by solving for fixed points of a differentiable function and using the last iterate of fixed point solver to backpropagate. In practice, we apply our model on several vision and language downstream tasks including visual question answering, visual reasoning, and image-text retrieval on different datasets. Our experiments show that it is possible to design models that perform similarly to state-of-art results but with significantly fewer samples and less training time.

<img width="1893" alt="pipeline2" src="https://github.com/ellenzhuwang/implicit_vkood/assets/10067151/9bdd6449-38d8-4269-9382-0dcc3395c561">


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
