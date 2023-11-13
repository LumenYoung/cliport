export CLIPORT_ROOT=$(pwd)

# python3 cliport/demos.py n=10 \
#                         task=assembling-kits-seq-unseen-colors \
#                         mode=test
# python3 cliport/demos.py n=10 \
#                         task=assembling-kits-seq-full-colors \
#                         mode=test
# python3 cliport/demos.py n=10 \
#                         task=put-block-in-bowl-full \
#                         mode=test
# python3 cliport/demos.py n=10 \
#                         task=assembling-kits-seq-full \
#                         mode=test
# python3 cliport/demos.py n=10 \
#                         task=towers-of-hanoi-seq-full \
#                         mode=test


# python3 cliport/demos.py n=10 \
#                         task=assembling-kits-seq-seen-colors \
#                         mode=test

# python3 cliport/eval.py eval_task=assembling-kits-seq-seen-colors \
#                        model_task=multi-language-conditioned \
#                        agent=cliport \
#                        mode=test \
#                        n_demos=10 \
#                        train_demos=1000 \
#                        exp_folder=cliport_quickstart \
#                        checkpoint_type=test_best \
#                        update_results=True \
#                        disp=False

# python3 cliport/eval.py eval_task=assembling-kits-seq-unseen-colors \
#                        model_task=multi-language-conditioned \
#                        agent=cliport \
#                        mode=test \
#                        n_demos=10 \
#                        train_demos=1000 \
#                        exp_folder=cliport_quickstart \
#                        checkpoint_type=test_best \
#                        update_results=True \
#                        disp=False

# python3 cliport/eval.py eval_task=assembling-kits-seq-full-colors \
#                        model_task=multi-language-conditioned  \
#                        agent=cliport \
#                        agent=cliport \
#                        mode=test \
#                        n_demos=10 \
#                        train_demos=1000 \
#                        exp_folder=cliport_quickstart \
#                        checkpoint_type=test_best \
#                        update_results=True \
#                        disp=False

# python3 cliport/eval.py eval_task=put-block-in-bowl-full \
#                        model_task=multi-language-conditioned \
#                        agent=cliport \
#                        mode=test \
#                        n_demos=10 \
#                        train_demos=1000 \
#                        exp_folder=cliport_quickstart \
#                        checkpoint_type=test_best \
#                        update_results=True \
#                        disp=False

# python3 cliport/eval.py eval_task=assembling-kits-seq-full \
#                        model_task=multi-language-conditioned \
#                        agent=cliport \
#                        mode=test \
#                        n_demos=10 \
#                        train_demos=1000 \
#                        exp_folder=cliport_quickstart \
#                        checkpoint_type=test_best \
#                        update_results=True \
#                        disp=False

# python3 cliport/eval.py eval_task= towers-of-hanoi-seq-full \
#                        model_task=multi-language-conditioned \
#                        agent=cliport \
#                        mode=test \
#                        n_demos=10 \
#                        train_demos=1000 \
#                        exp_folder=cliport_quickstart \
#                        checkpoint_type=test_best \
#                        update_results=True \
#                        disp=False


# python3 cliport/eval.py eval_task=block-insertion \
#                        model_task=multi-language-conditioned \
#                        agent=cliport \
#                        mode=test \
#                        n_demos=10 \
#                        train_demos=1000 \
#                        exp_folder=cliport_quickstart \
#                        checkpoint_type=test_best \
#                        update_results=True \
#                        disp=False

function get_and_run_cliport() {
  local eval_task=$1

  python3 cliport/demos.py n=10 \
                          task=${eval_task} \
                          mode=test

  python3 cliport/eval.py eval_task=${eval_task} \
                         model_task=multi-language-conditioned \
                         agent=cliport \
                         mode=test \
                         n_demos=10 \
                         train_demos=1000 \
                         exp_folder=cliport_quickstart \
                         checkpoint_type=test_best \
                         update_results=True \
                         disp=False
}

function run_cliport() {
  local eval_task=$1

  python3 cliport/eval.py eval_task=${eval_task} \
                         model_task=multi-language-conditioned \
                         agent=cliport \
                         mode=test \
                         n_demos=10 \
                         train_demos=1000 \
                         exp_folder=cliport_quickstart \
                         checkpoint_type=test_best \
                         update_results=True \
                         disp=False
}

function run_cliport_controled() {
  local eval_task=$1

  python3 cliport/eval.py eval_task=${eval_task} \
                         model_task=multi-language-conditioned \
                         agent=cliport \
                         mode=test \
                         n_demos=10 \
                         train_demos=1000 \
                         exp_folder=cliport_controled_exp \
                         checkpoint_type=test_best \
                         update_results=True \
                         disp=False
}

# Define an array of task names
# task_names=("block-insertion" "palletizing-boxes" "sweeping-piles" "put-block-in-bowl-unseen-colors")
# task_names=("palletizing-boxes" "block-insertion" "sweeping-piles" "put-block-in-bowl-unseen-colors")

# done testing, failing because it can't associate the color with the corresponding word.
# task_names=("assembling-kits-seq-unseen-colors")

# done testing, feedback on color can improve picking but placing is completely non-sense

# Works fine without feedback
# task_names=("put-block-in-bowl-seen-colors")

# Some color indication works fine
# task_names=("sweeping-piles")

task_names=("separating-piles-full")

# hard to feedback
# task_names=("put-block-in-bowl-unseen-colors")

# stack block pyramid
task_names=("stack-block-pyramid-seq-seen-colors")

task_names=("towers-of-hanoi-seq-full")

task_names=("palletizing-boxes")

task_names=("block-insertion")
# Loop over the array
for task in "${task_names[@]}"
do
  # run_cliport $task
  run_cliport_controled $task
done
