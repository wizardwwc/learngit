# PhySense

This repository hosts the source code for the paper "[PhySense: Defending Physically Realizable Attacks for Autonomous Systems via Consistency Reasoning](https://zh1yu4nyu.github.io/files/ZhiyuanYu_CCS24_PhySense.pdf)". The paper has been accepted by [the 31st ACM Conference on Computer and Communications Security (CCS), 14-18 October 2024](https://www.sigsac.org/ccs/CCS2024).

PhySense is a defense-in-depth system against physical realizable adversarial attacks. The key approach relies on reasoning, empowered by statistical modeling, robust physical rules, and pipelining techniques to ensure reliable and timely defense. PhySense not only detects malicious objects but also provides the potential true labels to correct misclassifications.

If you find our project useful, please cite us at:

```
@inproceedings{yu2024physense,
  title={PhySense: Defending Physically Realizable Attacks for Autonomous Systems via Consistency Reasoning},
  author={Yu, Zhiyuan and Li, Ao and Wen, Ruoyao and Chen, Yijia and Zhang, Ning},
  booktitle={Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security},
  year={2024}
}
```

## Hardware Requirements

The artifact can run on a machine with a moderate CPU and a GPU with at least 12 GB of available VRAM. In our setup, the experiments were conducted on a server with RTX 4090 GPU (24 GB VRAM) and Intel i9-13900K. Please note that runtime may vary depending on the hardware. Due to the large scale of the datasets used in our evaluation, a minimum of 1 TB of storage space is required.

## File Organizations

Due to the file size limit, some of the files involved in experiments need to be downloaded from the external [drive](https://wustl.box.com/s/bc2ts9smqj3otof76fgru7pcx944mx1p), which contains our collected data and pre-trained model weights. The layout of the entire artifact is as follows:

- Content in this Repository

```
.
|-- adaptive_attack
|   |-- ...
|-- data (Stored on the Drive)
|   |-- ...
|-- object_track
|   |-- ...
|-- phySense_initial
|   |-- ...
|-- phySense
|   |-- ...
|-- scripts
|   |-- ...
|-- util
|-- |-- ...
|-- weights (Stored on the Drive)
|-- |-- ...
|-- evaluate_carla_largepatch.py
|-- evaluate_carla_smallpatch.py
|-- evaluate_kitti_largepatch.py
|-- evaluate_kitti_smallpatch.py
|-- evaluate_nusc_largepatch.py
|-- evaluate_nusc_smallpatch.py
|-- LICENSE
|-- requirements.txt
`-- README.md
```

`phySense_initial` contains the initial implementation of our proposed PhySense system. Using this version, we analyzed its runtime breakdown to understand the performance bottleneck. This analysis is also included for reproducibility, please see `Initial Runtime Breakdown` for more details.  

`phySense` contains our final design, with the additional pipelined optimization for improved efficiency. This is used for the main evaluation of our system. 

- Content on the Drive

```
.
|-- Evaluation Data
|   |-- data.tar*
|   |-- weights.zip
|-- PhySense Carla Dataset
`-- |-- physense_carla.tar*
```

To reproduce our results, please place the content of `Evaluation data/weights` and `Evaluation data/data` under the main directory. To facilitate research in the field, we also provide the raw dataset collected from the Carla simulator in the `PhySense Carla Dataset` folder, whch is not required for the following experiments.

## Installation and Setup

PhySense was implemented using PyTorch, and the environment was tested using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 23.10.0 on Ubuntu 22.04.6 LTS. It is recommended to use CUDA version 11.3 or higher. The commands for setting up the environment are:

```sh
$ conda create -n physense python=3.8
$ conda activate physense
$ conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
```

## Basic Test

The basic test checks the existence of all required dependencies and files needed, as well as a simple functional check of the key modules of PhySense. To run the basic test:

```sh
$ python scripts/basic_test.py
```

The expected outputs include detailed information about the individual components such as package versions, as well as any error messages that may occur during checking.

## Behavior Inspection & Human Annotation (1 minute execution time, 1 GB Disk)

We developed annotated behavioral data built upon existing datasets. To do so, we implemented an interactive HTML-based visualization showing 3D bounding boxes with object labels and tracking IDs across frames. This corresponds to Section 5.5 (Characterization via Object Behaviors) in our paper.

- The command to generate interactive HTMLs is:

```sh
python scripts/scenes_animation.py
```

The expected outputs are a list of HTML files stored in the `results/scenes_html` folder, and we have included our pre-computed ones in the folder. Each HTML shows the 3D bounding boxes of objects along the time dimension, enabling human inspection and annotation of object behaviors.

- Based on these interactive HTMLs, human annotators labeled object behaviors, and the annotations are stored in the `./data/nusc/nuscenes_behavior_dataset.pkl` file. To inspect it, the following command will print out the size of our annotated behavior dataset and the label space.

```sh
python scripts/show_behavior_anno.py
```

The first line of the printed output shows the number of annotated data points, which is 13857, consistent with the number reported in Section 5.5 (Characterization via Object Behaviors) of our manuscript. The subsequent lines provide a detailed breakdown of the count for each behavior label, as well as an example of the extracted behavior data.

## Behavior Model Training (5 minutes execution time, 1 GB disk)

With the annotated behavior data, the behavior model can be trained using the following command, which includes optional arguments:

```sh
$ python scripts/train_behavior.py \
  --cuda_device "0" \
  --seed 42 \
  --dataset_file data/nusc/nuscenes_behavior_dataset.pkl \
  --train_frac 0.7 \
  --val_frac 0.15 \
  --test_frac 0.15 \
  --input_size 9 \
  --hidden_size 512 \
  --num_layers 3 \
  --num_epochs 300 \
  --learning_rate 0.001 \
  --batch_size 32 \
  --model_save_path \
  trained_models/lstm_attn_model.pth \
  --label_to_int_save_path \
  trained_models/lstm_label_to_int.pkl
```

Alternatively, the model can be trained with default arguments/parameters using the command:

```sh
$ python scripts/train_behavior.py 
```

The expected results from executing either command include the numerical behavior labels saved in `trained_models/lstm_label_to_int.pkl` and the trained behavior model stored in `trained_models/lstm_attn_model.pth`. Our pre-trained models are also provided in the `weights` folder. Upon completion of the training, the script will automatically perform validation and print out the validation accuracy, including overall accuracy and per-class performance (precision, recall, F1 scores), in the terminal output.

## Interaction Inference (1 minute execution time, 1 GB Disk)

We compiled a list of rules to detect the existence of interactions between objects and identify the potential interaction types. These rules are implemented in `util/interaction_class_inference.py`, corresponding to Section 5.6 (Characterization via Interactions) and Table 4 in the appendix of the paper. To apply these rules to extract potential interactions on existing datasets:

```sh
$ python scripts/interactions_construct_nusc.py
$ python scripts/interactions_construct_kitti.py
$ python scripts/interactions_construct_carla.py
```

This will produce the extracted interactions stored in pickle files under the path `data/{dataset}/crf_dataset_train`, where {dataset} can be `nusc`, `kitti`, and `carla`. Our generated files are provided in this folder as well. Upon executing the script, an example will be printed, showing the ground truth labels of a pair of two objects and the labels identified through potential interactions. 

## Initial Runtime Breakdown (3 minutes execution time, 1 GB disk)

To investigate the bottleneck of run-time efficiency, we first analyzed the run-time breakdown of the initial naive implementation of PhySense (`phySense_initial`), corresponding to Section 5.8 (Optimizing Run Time Efficiency) and Table 1 in our manuscript. The command for the analysis is:

```sh
$ python scripts/initial_runtime.py
```

The script will print the execution time for the main stages of PhySense, measured in seconds. The reported times should match those reported in Table 1 of our manuscript. The results indicate that the majority of computational delays are CPU-bound. 

## Physical Realizable Attacks (10 hours execution time, 20 GB Disk)

To evaluate defenses, we began with implementing attacks following existing studies. We adapted the adversarial attacks to our context (i.e., misclassification by object tracking systems), and applied transformations to simulate physical-world conditions such as angular positions and occlusions.

- The full command to optimize and apply adversarial patterns with optional arguments is:

```sh
$ python scripts/physical_attack.py \
  --device "cuda:0" \
  --seed 42 \
  --num_epochs 50 \
  --batch_size 32 \
  --adam_lr 0.01 \
  --patch_size [small/large] \
  --dataset [nuscenes/kitti]
```

The generated adversarial patterns and attack information (adversarial loss values) will be stored in the `attacker` folder. Our generated samples used in the evaluation are provided on cloud storage.

Alternatively, the attack can be executed with default arguments/parameters:

```sh
$ python scripts/physical_attack.py \
  --patch_size [small/large] \
  --dataset [nuscenes/kitti]
```

- The attack effectiveness can be verified with our target object tracking model, YOLOv3 combined with the SORT algorithm. The example command is:

```sh
$ python object_track/yolo_sort.py
```

The tracking results are object labels, which are crosschecked against the corresponding ground truth labels. Any mismatch is considered a misclassification and is counted as a successfully attacked frame. The expected output is the average number of frames and attack successful rate.

## Evaluation of PhySense (30 hours execution time, 1 TB disk)

The efficacy of PhySense can be evaluated using the implemented attacks. The experiments involves different attack methods, patch sizes, and diverse datasets. To evaluate PhySense on the nuScenes dataset with diverse patch sizes:

```sh
$ python evaluate_nusc_largepatch.py
$ python evaluate_nusc_smallpatch.py
```

Similarly, to evaluate against the KITTI dataset:

```sh
$ python evaluate_kitti_largepatch.py
$ python evaluate_kitti_smallpatch.py
```

To evaluate against our custom Carla dataset:

```sh
$ python evaluate_carla_largepatch.py
$ python evaluate_carla_smallpatch.py
```

The results are measured in terms of detection accuracy, correction accuracy, false positive rate (FPR), false negative rate (FNR), and average run-time. The results should be consistent with those reported in Table 2 in our manuscript.

## Adaptive Attack (50 hours execution time, 1 TB disk)

To understand the defense performance when facing adaptive attackers, we implemented three adaptive attack strategies: disrupt texture and behavior feature analysis, use the direct output of PhySense to guide optimization, and attack the temporal graph component. More details of these strategies can be found in Section 6.6 (PhySense against Adaptive Attackers) of our manuscript. 

- To run the attack using the first strategy:

```sh
$ python adaptive_attack/texture_behavior.py
```

To run the attack using the second strategy:

```sh
$ python adaptive_attack/defenseoutput.py
```

To run the attack using the third strategy:

```sh
$ python adaptive_attack/temporalgraph.py
```

- The attack effectiveness can be evaluated by:

```sh
$ python adaptive_attack/attack_evaluation.py
```

To reproduce our reported attack success rate using our generated adaptive adversarial patterns:

```sh
$ python adaptive_attack/attack_reproduced.py
```

The expected output is the average attack success rate of the optimized adaptive adversarial patterns bypassing our PhySense defense. The results produced by `adaptive_attack/attack_reproduced.py` should be consistent with those reported in Section 6.6 (PhySense against Adaptive Attackers) of our manuscript. 
