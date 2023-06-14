################################################################################
# PUBLIC VARIABLES
################################################################################

# The dataset to train the model on, either "CIFAR10" or "CIFAR100"
DATASET := CIFAR10

# The batch size to train with
BATCH_SIZE := 128

# The number of subprocesses to use for data loading
WORKERS := $(shell expr $(shell nproc) / 2)

# The number of epochs to run.
EPOCHS := 350

# The directory to write results to
OUT_DIR := out

################################################################################
# PRIVATE VARIABLES
################################################################################

_NUM_CLASSES := $(subst CIFAR,,$(DATASET))
_LABEL_SEQ := $(shell seq 0 $(shell expr $(_NUM_CLASSES) - 1))

_MODEL_DIR := $(OUT_DIR)/models
_LOG_DIR := $(OUT_DIR)/logs
_DATASET_DIR := $(OUT_DIR)/datasets
_ACTIVATIONS_DIR := $(OUT_DIR)/activations

_MODEL_PRE := $(DATASET)_ResNet18_Custom_Aug
_MODEL := $(_MODEL_DIR)/$(_MODEL_PRE)/best.pth
_FULL_ACTIVATIONS := $(foreach i, $(_LABEL_SEQ),\
	$(_ACTIVATIONS_DIR)/$(_MODEL_PRE)/full_activations/label$i.hdf5)
_SAMPLED_ACTIVATIONS := $(foreach i, $(_LABEL_SEQ),\
	$(_ACTIVATIONS_DIR)/$(_MODEL_PRE)/sampled_activations/label$i.hdf5)

################################################################################
# COMMANDS
################################################################################

.PHONY: train full clean sampled

train: $(_MODEL)

full: $(_FULL_ACTIVATIONS)

sampled: $(_SAMPLED_ACTIVATIONS)

clean:
	@rm -rf $(OUT_DIR)

################################################################################
# RULES
################################################################################

$(OUT_DIR):
	@mkdir -pv $@
	@echo "*" > $@/.gitignore

$(_MODEL): | $(OUT_DIR)
	@echo "> Training model"
	@python cifar_train.py \
		--dataset $(DATASET) \
		--batch-size $(BATCH_SIZE) \
		--workers $(WORKERS) \
		--epochs $(EPOCHS) \
		--log-dir $(_LOG_DIR) \
		--dataset-dir $(_DATASET_DIR) \
		--model-dir $(_MODEL_DIR)

$(_FULL_ACTIVATIONS): $(_MODEL) | $(OUT_DIR)
	@echo "> Extracting full activation vectors"
	@python cifar_extract_full_activations.py \
		--dataset $(DATASET) \
		--batch-size $(BATCH_SIZE) \
		--workers $(WORKERS) \
		--dataset-dir $(_DATASET_DIR) \
		--model-dir $(_MODEL_DIR) \
		--act-dir $(_ACTIVATIONS_DIR)

$(_SAMPLED_ACTIVATIONS): $(_MODEL) | $(OUT_DIR)
	@echo "> Extracting sampled activation vectors"
	@python cifar_extract_sampled_activations.py \
		--dataset $(DATASET) \
		--batch-size $(BATCH_SIZE) \
		--workers $(WORKERS) \
		--dataset-dir $(_DATASET_DIR) \
		--model-dir $(_MODEL_DIR) \
		--act-dir $(_ACTIVATIONS_DIR)