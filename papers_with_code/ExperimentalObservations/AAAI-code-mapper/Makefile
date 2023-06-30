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
INTERVAL		:= 40
OVERLAP			:= 30

################################################################################
# PRIVATE VARIABLES
# -----------------
# 
################################################################################

_CIFAR10_LAYERS := bn1 \
	layer1.0.bn1 layer1.0.bn2 layer1.1.bn1 layer1.1.bn2 \
	layer2.0.bn1 layer2.0.bn2 layer2.1.bn1 layer2.1.bn2 \
	layer3.0.bn1 layer3.0.bn2 layer3.1.bn1 layer3.1.bn2 \
	layer4.0.bn1 layer4.0.bn2 layer4.1.bn1 layer4.1.bn2


_ACTIVATIONS_DIR 	:= $(OUT_DIR)/activations/$(DATASET)_ResNet18_Custom_Aug
_MAPPER_DIR				:= $(OUT_DIR)/mapper_graphs/$(DATASET)_ResNet18_Custom_Aug
_FULL_MAPPERS_DIR := $(_MAPPER_DIR)/full_batches

_MODEL 						:= $(OUT_DIR)/models/$(DATASET)_ResNet18_Custom_Aug/touch_$(EPOCHS)
_FULL_ACTIVATIONS := $(_ACTIVATIONS_DIR)/full_activations/touch
_FULL_MAPPERS			:= $(foreach l, $(_CIFAR10_LAYERS),\
	$(_FULL_MAPPERS_DIR)/touch_$(l)_$(INTERVAL)_$(OVERLAP))

################################################################################
# COMMANDS
# --------
# 
################################################################################

.PHONY: clean train full

train: $(_MODEL)

full: $(_FULL_MAPPERS)

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
	@touch $@

$(_FULL_ACTIVATIONS): $(_MODEL)
	@echo "---- Extracting full activations ----"
	@python cifar_extract_full_activations.py \
		--cache-dir $(OUT_DIR) \
		--dataset $(DATASET) \
		--workers $(WORKERS) 
	@touch $@

$(_FULL_MAPPERS_DIR)/touch_%_$(INTERVAL)_$(OVERLAP): $(_FULL_ACTIVATIONS)
	@echo "---- Generating mapper graph for layer $(*F) ----"
	@python get_mapper_full_batches.py $(*F)\
		--cache-dir $(OUT_DIR) \
		--dataset $(DATASET) \
		--interval $(INTERVAL) \
		--overlap $(OVERLAP)
	@touch $@
	



