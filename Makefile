docker:
	docker buildx build --platform=linux/amd64 \
	 -f Dockerfile_nvidia -t l2cs-net_nvidia:$(shell poetry version -s) \
	 --cache-from=type=inline \
	 --cache-to=type=inline . --load
