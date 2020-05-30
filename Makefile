#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL = /bin/sh
PYTHON = python

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = titanic
KAGGLE_COMP = titanic

DATA_RAW = data/raw
DATA_INTERIM = data/interim
DATA_PROCESSED = data/processed
MODELS_FITTED = models/fitted
MODELS_PREDICTIONS = models/predictions
# Use the kaggle API to get a list of competition files.
# Then parse those filenames and append the raw data directory.
# Used to define prerequisites for the "make raw_data" rule.
RAW_FILES = $(shell kaggle competitions files -v $(KAGGLE_COMP) | \
				tail -n +2 | \
				cut -d "," -f1 | \
				while read line; do echo "$(DATA_RAW)/$${line}"; done | \
				tr \\n " ")

PROCESSED_FILES = $(DATA_PROCESSED)/train.csv $(DATA_PROCESSED)/test.csv

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: clean mostlyclean docs githooks raw_data data models predictions

## Set up githooks
githooks:
	@echo "Making files in .githooks/ executable..."
	@chmod +x .githooks/*
	@echo "Setting git hook path to .githooks..."
	@git config core.hooksPath .githooks
	@echo "Done."

## Build documentation and copy it to the docs folder
docs: docsrc/source
	@echo "Making documentation..."
	@(cd docsrc && make github)
	@echo "Done."

docsrc/source: docsrc/Makefile
	@sphinx-apidoc -o source/ ../$(PROJECT_NAME)

## Delete all models and data
clean:
	@echo "Removing predictions..."
	@rm -rf $(MODELS_PREDICTIONS)/*
	@echo "Removing models..."
	@rm -rf $(MODELS_FITTED)/*
	@echo "Removing processed data..."
	@rm -rf $(DATA_PROCESSED)/*
	@echo "Removing interim data..."
	@rm -rf $(DATA_INTERIM)/*
	@echo "Removing raw data..."
	@rm -rf $(DATA_RAW)/*
	@echo "Done."

## Delete all models and data, except for the raw downloads
mostlyclean:
	@echo "Removing predictions..."
	@rm -rf $(MODELS_PREDICTIONS)/*
	@echo "Removing models..."
	@rm -rf $(MODELS_FITTED)/*
	@echo "Removing processed data..."
	@rm -rf $(DATA_PROCESSED)/*
	@echo "Removing interim data..."
	@rm -rf $(DATA_INTERIM)/*
	@echo "Done."

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Pull raw data from Kaggle
raw_data: $(RAW_FILES)
## Transform raw data
data: $(PROCESSED_FILES)
## Train models
models: $(MODELS_FITTED)/*.joblib
## Create predictions
predictions: $(MODELS_PREDICTIONS)/*.csv

$(RAW_FILES) &: $(DATA_RAW)/titanic.zip
	@unzip -DD -u $(DATA_RAW)/$(KAGGLE_COMP).zip -d $(DATA_RAW)

$(DATA_RAW)/titanic.zip:
	@kaggle competitions download --path $(DATA_RAW) $(KAGGLE_COMP)

$(PROCESSED_FILES) &: $(DATA_INTERIM)/all.pkl.zip
	@$(PYTHON) titanic/features/transform.py $(DATA_INTERIM) $(DATA_PROCESSED)

$(DATA_INTERIM)/all.pkl.zip: $(RAW_FILES)
	@$(PYTHON) titanic/data/clean.py $(DATA_RAW) $(DATA_INTERIM)

$(MODELS_FITTED)/*.joblib: $(DATA_PROCESSED)/train.csv
	@$(PYTHON) titanic/models/train.py $(DATA_PROCESSED) $(MODELS_FITTED)

$(MODELS_PREDICTIONS)/*.csv: $(DATA_PROCESSED)/test.csv $(MODELS_FITTED)/*.joblib
	@$(PYTHON) titanic/models/predict.py $(DATA_PROCESSED) $(MODELS_FITTED) $(MODELS_PREDICTIONS)


#################################################################################
# SELF DOCUMENTING COMMANDS                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')