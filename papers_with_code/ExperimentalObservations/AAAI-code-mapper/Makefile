################################################################################
# PUBLIC VARIABLES
# ----------------
# 
################################################################################

OUT_DIR 		:= out
DATASET 		:= CIFAR10
# TODO: Change to 350
EPOCHS			:= 1
BATCH_SIZE	:= 128
WORKERS			:= $(shell expr $(shell nproc) / 2)

################################################################################
# PRIVATE VARIABLES
# -----------------
# 
################################################################################

_MODEL := $(OUT_DIR)/models/$(DATASET)_ResNet18_Custom_Aug/best.pth

################################################################################
# COMMANDS
# --------
# 
################################################################################

.PHONY: clean train

train: $(_MODEL)

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
		--batch-size $(BATCH_SIZE) \
		--workers $(WORKERS) \
		--epochs $(EPOCHS)




