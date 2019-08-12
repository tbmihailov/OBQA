## Set environment
# Environment Setup Instruction

1. Create the `careful-obqa` environment using Anaconda

  ```
  conda create -n careful-obqa python=3.6
  ```

2. Activate the environment

  ```
  source activate careful-obqa
  ```

3. Install the requirements in the environment:

  ```
  pip install -r requirements.txt
  ```


## Prepare data

Create data directory with the data from https://drive.google.com/drive/folders/1XM9krxl7weUITTIAIVRZXPSOKodn4Ixu?usp=sharing
(the same structure).

```
export PREPARED_DATA=~/research/data/obqa_careful_prepared
mkdir -p ${PREPARED_DATA}/generated
```

Create dirs
```
mkdir -p output
mkdir -p logs
```
## Run experiments

### OpenBookQA

#### Train

Bert OBQA classifier

```
job_name_base=obqa-bert
job_name_specific=${job_name_base}-gold-f2

PREPARED_DATA_DIR=~/research/data/obqa_careful_prepared

#params
mode=train
#mode=predict

output_model_dir=AUTO
#output_model_dir=output/bertqa-obqa-bert-gold-2-mergeFalse-goldFalse-128-bak/

input_file_dev=hyp-gold-val.tsv
input_file_test=hyp-gold-test.tsv
knowledge_corpus_file=/home/mitarb/mihaylov/research/data/obqa_careful_prepared/knowledge/openbook.txt
knowledge_ranking_file=/home/mitarb/mihaylov/research/data/obqa_careful_prepared/ranked/sts-trained-openbook.json
#knowledge_ranking_file=/home/mitarb/paul/openqa/fact_1_graph_based_dev_test.json

gold_flag=--gold
#gold_flag=

method=sum
topk=2

echo "$input_file"
JOB_NAME=${job_name_specific}

job_mem=60G

job_time=71:59:00   # HH:mm:ss

job_cpus_per_task=4

gpu_selector=gpu:mem11g
# gpu:1/gpu:mem8g:1/gpu:mem11g:2

# params below not changed very often

# CHANGEME - path to your environment python (when environment is activated write "which python")
python_exec=/home/mitarb/mihaylov/anaconda3/envs/careful-obqa/bin/python
JOB_NAME=${JOB_NAME}_$(date +%y-%m-%d-%H-%M-%S)

# CHANGEME - your script for running one single file.
JOB_SCRIPT="PREPARED_DATA=${PREPARED_DATA_DIR} PYTHONPATH=. $python_exec bert/run_trainqa.py ${gold_flag} --exp ${job_name_specific} --method ${method} --topk ${topk} --mode ${mode} --input_file_dev ${input_file_dev} --input_file_test ${input_file_test} --knowledge_corpus_file ${knowledge_corpus_file} --knowledge_ranking_file ${knowledge_ranking_file} --output_model_dir ${output_model_dir}"


# DO NOT change anything here!
echo "bash ~/research/cmd/hd-icl/run_sbatch.sh \"${JOB_SCRIPT}\" ${JOB_NAME} ${job_mem} ${job_time} ${job_cpus_per_task} ${gpu_selector}"
bash ~/research/cmd/hd-icl/run_sbatch.sh "${JOB_SCRIPT}" ${JOB_NAME} ${job_mem} ${job_time} ${job_cpus_per_task} ${gpu_selector}

```



#### EVAL

Bert OBQA classifier

```
job_name_base=obqa-bert
job_name_specific=${job_name_base}-predict-comet

PREPARED_DATA_DIR=~/research/data/obqa_careful_prepared

#params
mode=predict

output_model_dir=AUTO
output_model_dir=output/bertqa-obqa-bert-gold-2-mergeFalse-goldFalse-128-bak/

# gold
#input_file_dev=hyp-gold-val.tsv
#input_file_test=hyp-gold-test.tsv

# question only
input_file_dev=hyp-ques-val.tsv
input_file_test=hyp-ques-test.tsv

# IF you have a custom file - copy it to PREPARED_DATA_DIR/hypothesis/ and use only the base name
#input_file_dev=comet_test_fact_1_fact_2.tsv
#input_file_test=comet_test_fact_1_fact_2.tsv

knowledge_corpus_file=/home/mitarb/mihaylov/research/data/obqa_careful_prepared/knowledge/openbook.txt
knowledge_ranking_file=/home/mitarb/mihaylov/research/data/obqa_careful_prepared/ranked/sts-trained-openbook.json
#knowledge_ranking_file=/home/mitarb/paul/openqa/fact_1_graph_based_dev_test.json
#knowledge_ranking_file=/home/mitarb/paul/openqa/comet_test_fact_1_fact_2.tsv

gold_flag=--gold
#gold_flag=


method=sum
topk=0

echo "$input_file"
JOB_NAME=${job_name_specific}

job_mem=12G

job_time=71:59:00   # HH:mm:ss

job_cpus_per_task=4

gpu_selector=gpu:mem11g
# gpu:1/gpu:mem8g:1/gpu:mem11g:2

# params below not changed very often

# CHANGEME - path to your environment python (when environment is activated write "which python")
python_exec=/home/mitarb/mihaylov/anaconda3/envs/careful-obqa/bin/python
JOB_NAME=${JOB_NAME}_$(date +%y-%m-%d-%H-%M-%S)

# CHANGEME - your script for running one single file.
JOB_SCRIPT="PREPARED_DATA=${PREPARED_DATA_DIR} PYTHONPATH=. $python_exec bert/run_trainqa.py ${gold_flag} --exp ${job_name_specific} --method ${method} --topk ${topk} --mode ${mode} --input_file_dev ${input_file_dev} --input_file_test ${input_file_test} --knowledge_corpus_file ${knowledge_corpus_file} --knowledge_ranking_file ${knowledge_ranking_file} --output_model_dir ${output_model_dir}"


# DO NOT change anything here!
echo "bash ~/research/cmd/hd-icl/run_sbatch.sh \"${JOB_SCRIPT}\" ${JOB_NAME} ${job_mem} ${job_time} ${job_cpus_per_task} ${gpu_selector}"
bash ~/research/cmd/hd-icl/run_sbatch.sh "${JOB_SCRIPT}" ${JOB_NAME} ${job_mem} ${job_time} ${job_cpus_per_task} ${gpu_selector}

```
