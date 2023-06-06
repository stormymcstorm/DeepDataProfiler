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

_MODEL_DIR := $(OUT_DIR)/models
_LOG_DIR := $(OUT_DIR)/logs
_DATASET_DIR := $(OUT_DIR)/datasets

_MODEL := $(_MODEL_DIR)/$(DATASET)_ResNet18_Custom_Aug/best.pth

################################################################################
# COMMANDS
################################################################################

.PHONY: train clean

train: $(_MODEL)

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
