# Local-only convenience targets; HPC stays in iridis/*/job*.sh.

PYTHON ?= python
RUN_DIR ?= iridis/analyze-lncot+adm/run_0

# `make synthesis` -- cross-checkpoint plotting from rsync'd run_N/<tag>/.
.PHONY: synthesis
synthesis:
	$(PYTHON) -m analysis.synthesis --run-dir $(RUN_DIR) --output-dir $(RUN_DIR)

.PHONY: help
help:
	@echo "Targets:"
	@echo "  synthesis  Cross-checkpoint synthesis (local; reads RUN_DIR=$(RUN_DIR))"
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  PYTHON   Python interpreter (default: python)"
	@echo "  RUN_DIR  Run directory to synthesise (default: iridis/analyze-lncot+adm/run_0)"
