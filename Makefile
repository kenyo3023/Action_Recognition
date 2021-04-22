init:
	python3 -m pip install --user pipenv
	pipenv --python 3.8
	pipenv install --dev --skip-lock

ENDING_LINT = **/*.py **/*.json **/*.md
LIB_LINT = action_recognition
RUN_LINT = run_*.py
PYTHON_LINT = $(LIB_LINT) $(RUN_LINT)
PROJECT_NAME = $(shell basename $(CURDIR))

init_jupyter_kernel:
	pipenv run python -m ipykernel install --user --name=${PROJECT_NAME}

lint: linter_version ending flake8 pylint mypy

linter_version:
	pipenv run pip list | grep -P "(flake8|pyflakes|pycodestyle)"
	pipenv run pip list | grep -P "(pylint|astroid)"
	pipenv run pip list | grep -P "(mypy|typing|typed)"

ending:
	! grep -rHnP --include="*.py" --include="*.json" --include="*.md" "\x0D" ${PYTHON_LINT}

flake8:
	pipenv run flake8 ${PYTHON_LINT}

pylint:
	pipenv run pylint ${PYTHON_LINT}

clean_mypy:
	rm -rf .mypy_cache/

mypy: clean_mypy
	pipenv run mypy --ignore-missing-imports ${PYTHON_LINT}

clean: clean_mypy

run_preprocess:
	pipenv run python3 run_preprocess.py $(ARGS)

run_experiment:
	pipenv run python3 run_experiment.py $(ARGS)

run_inference:
	pipenv run python3 run_inference.py $(ARGS)

# EXP_PROGRAM=pipenv run python3 run_experiment.py
EXP_PROGRAM=pipenv run python3 run_config.py
# EXP_ARGS=--config configs/experiment.yaml --base mouse_video I3D balancecage origin10vid no_valid --group 10vid_I3D_bc_lr0.005 --lr 0.005
# EXP_ARGS=--config configs/experiment.yaml --base mouse_video I3D balancecage no_valid --group 5min_I3D_nv_bc_lr0.005 --lr 0.005
# EXP_ARGS=--config configs/experiment.yaml --base mouse_video I3D balancecage fpvid no_valid --group fp_I3D_nv_bc
# EXP_ARGS=--config configs/experiment.yaml --base mouse_video I3D balancecage no_valid --group 5min_I3D_nv_bc
EXP_ARGS=--config configs/experiment.yaml --base mouse_video I3D balancecage origin10vid --group 10vid_I3D
# EXP_ARGS=--config configs/experiment.yaml --base mouse resnet101 balancecage lr_scheduler no_valid --group 5min_res101_nv_bc_lr1e-4 --lr 0.0001 --batch_size 8
# EXP_ARGS=--config configs/experiment.yaml --base breakfast --group bf_I3D_finetune
# EXP_ARGS=--config configs/experiment.yaml --base mpii --group mpii_I3D_finetune
run_N_experiment:
	# $(EXP_PROGRAM) $(EXP_ARGS) --cage 0
	# $(EXP_PROGRAM) $(EXP_ARGS) --cage 1
	# $(EXP_PROGRAM) $(EXP_ARGS) --cage 2
	# $(EXP_PROGRAM) $(EXP_ARGS) --cage 3
	# $(EXP_PROGRAM) $(EXP_ARGS) --cage 4
	# $(EXP_PROGRAM) $(EXP_ARGS) --cage 5
	# $(EXP_PROGRAM) $(EXP_ARGS) --cage 6
	# $(EXP_PROGRAM) $(EXP_ARGS) --cage 7
	# $(EXP_PROGRAM) $(EXP_ARGS) --cage 8
	# $(EXP_PROGRAM) $(EXP_ARGS) --cage 9
	$(EXP_PROGRAM) $(EXP_ARGS) --cage -1
	# $(EXP_PROGRAM) $(EXP_ARGS) --split cv_0
	# $(EXP_PROGRAM) $(EXP_ARGS) --split cv_1
	# $(EXP_PROGRAM) $(EXP_ARGS) --split cv_2
	# $(EXP_PROGRAM) $(EXP_ARGS) --split cv_3
	# $(EXP_PROGRAM) $(EXP_ARGS) --split cv_4
	# $(EXP_PROGRAM) $(EXP_ARGS) --split cv_5
	# $(EXP_PROGRAM) $(EXP_ARGS) --split cv_6

INF_PROGRAM=pipenv run python3 run_inference.py
INF_ARGS=--group inference_10vid_I3D_c0 --prob_thresh 0.5
run_N_inference:
	$(INF_PROGRAM) $(INF_ARGS) --run_id 25v2nlrs
	# $(INF_PROGRAM) $(INF_ARGS) --split 0 --run_id 2kvordd9
	# $(INF_PROGRAM) $(INF_ARGS) --split 1 --run_id 34puxld1
	# $(INF_PROGRAM) $(INF_ARGS) --split 2 --run_id 3adszh6j
	# $(INF_PROGRAM) $(INF_ARGS) --split 3 --run_id 2lv48dbd
	# $(INF_PROGRAM) $(INF_ARGS) --split 4 --run_id 26iy96g2
	# $(INF_PROGRAM) $(INF_ARGS) --split 5 --run_id 1ks9ixtv
	# $(INF_PROGRAM) $(INF_ARGS) --split 6 --run_id 2p1w7v2s
	# $(INF_PROGRAM) $(INF_ARGS) --split 7 --run_id 2gs8ohim
	# $(INF_PROGRAM) $(INF_ARGS) --split 8 --run_id zkj1qfml
	# $(INF_PROGRAM) $(INF_ARGS) --split 9 --run_id 3mzvt0ss

GINF_PROGRAM=pipenv run python run_group_inference.py --process_cnt 2 --random_split
run_group_inference:
	$(GINF_PROGRAM) --prob_thresh 0.5 --train_group 10vid_I3D
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group 5min_data_3D_balancevideo
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group fpvid_I3D_bv
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group 5min_I3D_exclude2mouse_bc
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group fpvid_I3D_bc
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group 5min_I3D_exclude2mousevd_bc
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group fpvid_I3D_exclude2mousevd_bc
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group 10vid_exclude2mousevd_bc
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group 5min_I3D_exclude2mousevd_bc
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group 10vid_I3D_exclude2mousevd_nv_bc
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group 5min_I3D_exclude2mousevd_nv_bc
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group 10vid_I3D_nv_bc
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group 5min_I3D_nv_bc
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group 5min_res101_nv_bc_lr1e-4
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group 10vid_I3D_bc_lr0.005
	# $(GINF_PROGRAM) --prob_thresh 0.5 --train_group 

WANDB_DIR="./output/wandb"
cfg="sweep.yaml"
set_sweep:
	WANDB_DIR=$(WANDB_DIR) pipenv run wandb sweep --project ${PROJECT_NAME} ${cfg}

run_sweep_agent:
	WANDB_DIR=$(WANDB_DIR) pipenv run wandb agent --project ${PROJECT_NAME} -e donny --count 10 $(SWEEP_ID)
