# Local-only convenience targets; HPC stays in iridis/*/job*.sh.

PYTHON ?= python
RUN_DIR ?= iridis/analyze-lncot+adm/run_0

# `make synthesis` -- cross-checkpoint plotting from rsync'd run_N/<tag>/.
.PHONY: synthesis
synthesis:
	$(PYTHON) -m analysis.synthesis --run-dir $(RUN_DIR) --output-dir $(RUN_DIR)

# `make rq9b` -- assemble RQ9b ANOVA inputs from a counting-sweep
# RUN_DIR, then run analysis.counting_anova_rq9b on them.
#
# Two-step pipeline: (a) walk the counting RUN_DIR for per-cell
# summary.json + eval_summary_*.json files and emit one
# <variant>_seed_<N>.json per cell into the schema consumed by
# analysis.counting_anova_rq9b; (b) invoke the ANOVA on the assembled
# inputs. Override RQ9B_RUN_DIR on the command line:
#   make rq9b RQ9B_RUN_DIR=iridis/counting-sweep/run_3
RQ9B_RUN_DIR ?= iridis/counting-sweep/run_0
RQ9B_OUT ?= $(RQ9B_RUN_DIR)/rq9b
.PHONY: rq9b
rq9b:
	$(PYTHON) scripts/rq9b_assemble_inputs.py \
	    --run-dir $(RQ9B_RUN_DIR) \
	    --output-dir $(RQ9B_OUT)/inputs
	$(PYTHON) -m analysis.counting_anova_rq9b \
	    --results-dir $(RQ9B_OUT)/inputs \
	    --output-dir $(RQ9B_OUT)

.PHONY: help
help:
	@echo "Targets:"
	@echo "  synthesis  Cross-checkpoint synthesis (local; reads RUN_DIR=$(RUN_DIR))"
	@echo "  rq9b       Assemble RQ9b inputs from a counting-sweep RUN_DIR"
	@echo "             and run analysis.counting_anova_rq9b on them."
	@echo "             (reads RQ9B_RUN_DIR=$(RQ9B_RUN_DIR))"
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  PYTHON         Python interpreter (default: python)"
	@echo "  RUN_DIR        Run directory to synthesise"
	@echo "                 (default: iridis/analyze-lncot+adm/run_0)"
	@echo "  RQ9B_RUN_DIR   Counting-sweep run dir for rq9b"
	@echo "                 (default: iridis/counting-sweep/run_0)"
	@echo "  RQ9B_OUT       rq9b output dir"
	@echo "                 (default: \$$(RQ9B_RUN_DIR)/rq9b)"
