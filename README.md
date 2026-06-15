# drm-sched-bench

A benchmark suite for evaluating Linux DRM GPU scheduler policies: FIFO, fair,
and deadline. It drives a single GPU ring with controlled synthetic Vulkan
compute load and measures fairness, weight-proportional sharing, and dispatch
latency from kernel ftrace.

## Build

```sh
# builds vkload and spin.spv
make
```

## Run

```sh
# single client, manual:
./vkload --job-us 5000 --rate 0 --queue-depth 48 --output out.csv

# full suite across all three policies (root; writes results/<timestamp>/):
sudo bash run-scenarios.sh --duration 20000 --device 0
```
