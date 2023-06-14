################################################################################
# PUBLIC VARIABLES
# ----------------
# 
################################################################################

OUT_DIR 		:= out
DATASET 		:= CIFAR10
# TODO: Change to 350
EPOCHS			:= 1
WORKERS			:= $(shell expr $(shell nproc) / 2)

################################################################################
# PRIVATE VARIABLES
# -----------------
# 
################################################################################

_MODEL := $(OUT_DIR)/models/$(DATASET)_ResNet18_Custom_Aug/best.pth
_ACTIVATIONS_DIR := $(OUT_DIR)/activations/$(DATASET)_ResNet18_Custom_Aug
_FULL_ACTIVATIONS := $(_ACTIVATIONS_DIR)/full_activations/touch

################################################################################
# COMMANDS
# --------
# 
################################################################################

.PHONY: clean train full

train: $(_MODEL)

full: $(_FULL_ACTIVATIONS)

clean:
	@echo Removing $(OUT_DIR)
	@rm -rf $(OUT_DIR)

################################################################################
# RULES
# -----
# 
################################################################################

$(OUT_DIR):
	@mkdir -pv $@
	@echo "*" > $@/.gitignore

$(_MODEL): | $(OUT_DIR)
	@echo "---- Training model ----"
	@python cifar_train.py \
		--cache-dir $(OUT_DIR) \
		--dataset $(DATASET) \
		--workers $(WORKERS) \
		--epochs $(EPOCHS)

$(_FULL_ACTIVATIONS): $(_MODEL)
	@echo "---- Extracting full activations ----"
	@python cifar_extract_full_activations.py  \
		--cache-dir $(OUT_DIR) \
		--dataset $(DATASET) \
		--workers $(WORKERS)


