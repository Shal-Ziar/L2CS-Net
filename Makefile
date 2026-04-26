# ------------------------------------------------------------
# Declare the “phony” (non‑file) targets
.PHONY: amd nvidia docker-build

# ------------------------------------------------------------
# Private recipe that does the real work.
# It expects the variable GROUP to be set beforehand.
docker-build:
	@echo "Building Docker image for L2CS‑Net with group $(GROUP)"
	@docker buildx build \
		--platform=linux/amd64 \
		-f Dockerfile \
		-t l2cs-net_$(GROUP):$(shell poetry version -s) \
		--cache-from=type=inline \
		--cache-to=type=inline \
		--build-arg GROUP=$(GROUP) \
		. --load

# ------------------------------------------------------------
# Public façade targets – they set GROUP and then invoke the private one.
amd:   GROUP = amd
amd:   docker-build          # <-- amd depends on docker-build

nvidia: GROUP = nvidia
nvidia: docker-build        # <-- nvidia depends on docker-build