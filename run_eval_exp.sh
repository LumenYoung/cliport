export CLIPORT_ROOT=$(pwd)

# task names
# assembling-kits-seq-unseen-colors
# assembling-kits-seq-full-colors
# assembling-kits-seq-seen-colors
# put-block-in-bowl-full
# assembling-kits-seq-full
# towers-of-hanoi-seq-full
# put-block-in-bowl-full
# block-insertion

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

function get_and_run_cliport_controled() {
  local eval_task=$1

  python3 cliport/demos.py n=100 \
                          task=${eval_task} \
                          mode=test

  python3 cliport/eval.py eval_task=${eval_task} \
                         model_task=multi-language-conditioned \
                         agent=cliport \
                         mode=test \
                         n_demos=50 \
                         train_demos=1000 \
                         exp_folder=cliport_quickstart \
                         checkpoint_type=test_best \
                         update_results=True \
                         disp=False

}

function exp_compare_feedback_accuracy() {
  local eval_task=$1

  python3 cliport/demos.py n=20 \
                          task=${eval_task} \
                          mode=test

  python3 cliport/eval.py eval_task=${eval_task} \
                         model_task=multi-language-conditioned \
                         agent=cliport \
                         mode=test \
                         n_demos=3 \
                         train_demos=1000 \
                         exp_folder=cliport_quickstart \
                         checkpoint_type=test_best \
                         update_results=True \
                         disp=False \
                         correction_feedback_agent="llava"

  # rm -rf data/${eval_task}-test

  # python3 cliport/demos.py n=20 \
  #                         task=${eval_task} \
  #                         mode=test

  # python3 cliport/eval.py eval_task=${eval_task} \
  #                        model_task=multi-language-conditioned \
  #                        agent=cliport \
  #                        mode=test \
  #                        n_demos=20 \
  #                        train_demos=1000 \
  #                        exp_folder=cliport_quickstart \
  #                        checkpoint_type=test_best \
  #                        update_results=True \
  #                        disp=False \
  #                        correction_feedback_agent="llava"
}

function run_cliport_controled() {
  local eval_task=$1

  python3 cliport/eval.py eval_task=${eval_task} \
                         model_task=multi-language-conditioned \
                         agent=cliport \
                         mode=test \
                         n_demos=100 \
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


# history

task_names=("assembling-kits-seq-full")

task_names=("towers-of-hanoi-seq-full")

task_names=("packing-shapes", "separating-piles-full")

task_names=("towers-of-hanoi-seq-full")

task_names=("towers-of-hanoi-seq-full")

task_names=("palletizing-boxes")

# Loop over the array
for task in "${task_names[@]}"
do
  # run_cliport $task
  # run_cliport_controled $task
  exp_compare_feedback_accuracy $task
done

