/*
 * vkload — Vulkan compute benchmark for DRM scheduler evaluation.
 */

#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

/* ======================== Configuration ======================== */

struct config {
	uint32_t device_idx;
	uint32_t job_us;        /* target GPU job duration */
	uint32_t rate_hz;       /* submission rate (0 = continuous) */
	uint32_t duration_ms;   /* total benchmark duration */
	uint32_t queue_depth;   /* max in-flight submissions */
	const char *output;     /* output CSV file path (NULL = stdout) */
	int calibrate_only;
	VkQueueGlobalPriorityKHR global_priority; /* 0 = don't set */
	const char *sync_dir;   /* barrier directory (NULL = no sync) */
	uint32_t sync_count;    /* number of processes to wait for */
};

static struct config cfg = {
	.device_idx = 0,
	.job_us = 1000,
	.rate_hz = 0,
	.duration_ms = 10000,
	.queue_depth = 4,
	.output = NULL,
	.calibrate_only = 0,
	.global_priority = 0,
	.sync_dir = NULL,
	.sync_count = 0,
};

static VkQueueGlobalPriorityKHR parse_priority(const char *s)
{
	if (!strcmp(s, "low"))
		return VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR;
	if (!strcmp(s, "medium"))
		return VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
	if (!strcmp(s, "high"))
		return VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR;
	if (!strcmp(s, "realtime"))
		return VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR;
	fprintf(stderr, "Unknown priority '%s' (use: low, medium, high, realtime)\n", s);
	exit(1);
}

static const char *priority_name(VkQueueGlobalPriorityKHR p)
{
	switch (p) {
	case VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR:      return "low";
	case VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR:   return "medium";
	case VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR:     return "high";
	case VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR: return "realtime";
	default: return "default";
	}
}

/* ======================== Vulkan state ======================== */

struct vk_state {
	VkInstance instance;
	VkPhysicalDevice phys_dev;
	VkDevice device;
	VkQueue queue;
	uint32_t queue_family;
	VkCommandPool cmd_pool;
	VkDescriptorPool desc_pool;
	VkDescriptorSetLayout desc_layout;
	VkPipelineLayout pipe_layout;
	VkPipeline pipeline;
	VkDescriptorSet desc_set;
	VkBuffer buffer;
	VkDeviceMemory buffer_mem;
	VkQueryPool query_pool;
	float timestamp_period; /* ns per tick */

	/* Push constant: loop count controlling GPU duration */
	uint32_t loop_count;
};

/* ======================== Shader ======================== */

/*
 * SPIR-V for a trivial compute shader
 * Compiled with: glslangValidator -V -o spin.spv spin.comp
 */
static const uint32_t shader_spirv[] = {
	0x07230203, 0x00010000, 0x0008000b, 0x00000038,
	0x00000000, 0x00020011, 0x00000001, 0x0006000b,
	0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
	0x00000000, 0x0003000e, 0x00000000, 0x00000001,
	0x0006000f, 0x00000005, 0x00000004, 0x6e69616d,
	0x00000000, 0x0000000b, 0x00060010, 0x00000004,
	0x00000011, 0x00000040, 0x00000001, 0x00000001,
	0x00030003, 0x00000002, 0x000001c2, 0x00040005,
	0x00000004, 0x6e69616d, 0x00000000, 0x00030005,
	0x00000008, 0x00786469, 0x00080005, 0x0000000b,
	0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f,
	0x496e6f69, 0x00000044, 0x00030005, 0x00000010,
	0x006c6176, 0x00030005, 0x00000012, 0x00667542,
	0x00050006, 0x00000012, 0x00000000, 0x61746164,
	0x00000000, 0x00030005, 0x00000014, 0x00000000,
	0x00030005, 0x0000001b, 0x00000069, 0x00030005,
	0x00000022, 0x00004350, 0x00050006, 0x00000022,
	0x00000000, 0x706f6f6c, 0x00000073, 0x00030005,
	0x00000024, 0x00000000, 0x00040047, 0x0000000b,
	0x0000000b, 0x0000001c, 0x00040047, 0x00000011,
	0x00000006, 0x00000004, 0x00030047, 0x00000012,
	0x00000003, 0x00050048, 0x00000012, 0x00000000,
	0x00000023, 0x00000000, 0x00040047, 0x00000014,
	0x00000021, 0x00000000, 0x00040047, 0x00000014,
	0x00000022, 0x00000000, 0x00030047, 0x00000022,
	0x00000002, 0x00050048, 0x00000022, 0x00000000,
	0x00000023, 0x00000000, 0x00040047, 0x00000037,
	0x0000000b, 0x00000019, 0x00020013, 0x00000002,
	0x00030021, 0x00000003, 0x00000002, 0x00040015,
	0x00000006, 0x00000020, 0x00000000, 0x00040020,
	0x00000007, 0x00000007, 0x00000006, 0x00040017,
	0x00000009, 0x00000006, 0x00000003, 0x00040020,
	0x0000000a, 0x00000001, 0x00000009, 0x0004003b,
	0x0000000a, 0x0000000b, 0x00000001, 0x0004002b,
	0x00000006, 0x0000000c, 0x00000000, 0x00040020,
	0x0000000d, 0x00000001, 0x00000006, 0x0003001d,
	0x00000011, 0x00000006, 0x0003001e, 0x00000012,
	0x00000011, 0x00040020, 0x00000013, 0x00000002,
	0x00000012, 0x0004003b, 0x00000013, 0x00000014,
	0x00000002, 0x00040015, 0x00000015, 0x00000020,
	0x00000001, 0x0004002b, 0x00000015, 0x00000016,
	0x00000000, 0x00040020, 0x00000018, 0x00000002,
	0x00000006, 0x0003001e, 0x00000022, 0x00000006,
	0x00040020, 0x00000023, 0x00000009, 0x00000022,
	0x0004003b, 0x00000023, 0x00000024, 0x00000009,
	0x00040020, 0x00000025, 0x00000009, 0x00000006,
	0x00020014, 0x00000028, 0x0004002b, 0x00000006,
	0x0000002b, 0x41c64e6d, 0x0004002b, 0x00000006,
	0x0000002d, 0x00003039, 0x0004002b, 0x00000015,
	0x00000030, 0x00000001, 0x0004002b, 0x00000006,
	0x00000035, 0x00000040, 0x0004002b, 0x00000006,
	0x00000036, 0x00000001, 0x0006002c, 0x00000009,
	0x00000037, 0x00000035, 0x00000036, 0x00000036,
	0x00050036, 0x00000002, 0x00000004, 0x00000000,
	0x00000003, 0x000200f8, 0x00000005, 0x0004003b,
	0x00000007, 0x00000008, 0x00000007, 0x0004003b,
	0x00000007, 0x00000010, 0x00000007, 0x0004003b,
	0x00000007, 0x0000001b, 0x00000007, 0x00050041,
	0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c,
	0x0004003d, 0x00000006, 0x0000000f, 0x0000000e,
	0x0003003e, 0x00000008, 0x0000000f, 0x0004003d,
	0x00000006, 0x00000017, 0x00000008, 0x00060041,
	0x00000018, 0x00000019, 0x00000014, 0x00000016,
	0x00000017, 0x0004003d, 0x00000006, 0x0000001a,
	0x00000019, 0x0003003e, 0x00000010, 0x0000001a,
	0x0003003e, 0x0000001b, 0x0000000c, 0x000200f9,
	0x0000001c, 0x000200f8, 0x0000001c, 0x000400f6,
	0x0000001e, 0x0000001f, 0x00000000, 0x000200f9,
	0x00000020, 0x000200f8, 0x00000020, 0x0004003d,
	0x00000006, 0x00000021, 0x0000001b, 0x00050041,
	0x00000025, 0x00000026, 0x00000024, 0x00000016,
	0x0004003d, 0x00000006, 0x00000027, 0x00000026,
	0x000500b0, 0x00000028, 0x00000029, 0x00000021,
	0x00000027, 0x000400fa, 0x00000029, 0x0000001d,
	0x0000001e, 0x000200f8, 0x0000001d, 0x0004003d,
	0x00000006, 0x0000002a, 0x00000010, 0x00050084,
	0x00000006, 0x0000002c, 0x0000002a, 0x0000002b,
	0x00050080, 0x00000006, 0x0000002e, 0x0000002c,
	0x0000002d, 0x0003003e, 0x00000010, 0x0000002e,
	0x000200f9, 0x0000001f, 0x000200f8, 0x0000001f,
	0x0004003d, 0x00000006, 0x0000002f, 0x0000001b,
	0x00050080, 0x00000006, 0x00000031, 0x0000002f,
	0x00000030, 0x0003003e, 0x0000001b, 0x00000031,
	0x000200f9, 0x0000001c, 0x000200f8, 0x0000001e,
	0x0004003d, 0x00000006, 0x00000032, 0x00000008,
	0x0004003d, 0x00000006, 0x00000033, 0x00000010,
	0x00060041, 0x00000018, 0x00000034, 0x00000014,
	0x00000016, 0x00000032, 0x0003003e, 0x00000034,
	0x00000033, 0x000100fd, 0x00010038,
};

/* ======================== Time helpers ======================== */

static uint64_t now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static void sleep_until_ns(uint64_t target)
{
	struct timespec ts = {
		.tv_sec = target / 1000000000ULL,
		.tv_nsec = target % 1000000000ULL,
	};

	while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR)
		;
}

/* ======================== Sync barrier ======================== */

static int count_ready_files(const char *dir)
{
	DIR *d = opendir(dir);
	int count = 0;

	if (!d)
		return 0;
	struct dirent *ent;
	while ((ent = readdir(d))) {
		size_t len = strlen(ent->d_name);
		if (len > 6 && !strcmp(ent->d_name + len - 6, ".ready"))
			count++;
	}
	closedir(d);
	return count;
}

/*
 * File-based barrier: each process touches <sync_dir>/<pid>.ready, then
 * spins until sync_count .ready files exist. Timeout after 30s.
 */
static void sync_barrier(void)
{
	char path[256];

	if (!cfg.sync_dir || cfg.sync_count < 2)
		return;

	snprintf(path, sizeof(path), "%s/%d.ready", cfg.sync_dir, getpid());
	FILE *f = fopen(path, "w");
	if (!f) {
		perror("sync barrier: fopen");
		exit(1);
	}
	fclose(f);

	fprintf(stderr, "Waiting for %u processes at barrier...\n", cfg.sync_count);

	uint64_t deadline = now_ns() + 30ULL * 1000000000ULL;
	while (now_ns() < deadline) {
		if ((uint32_t)count_ready_files(cfg.sync_dir) >= cfg.sync_count)
			break;
		usleep(1000); /* 1ms poll */
	}

	if ((uint32_t)count_ready_files(cfg.sync_dir) < cfg.sync_count) {
		fprintf(stderr, "Sync barrier timeout (got %d/%u)\n",
			count_ready_files(cfg.sync_dir), cfg.sync_count);
		exit(1);
	}

	fprintf(stderr, "Barrier passed, starting benchmark\n");
}

/* ======================== Vulkan helpers ======================== */

#define VK_CHECK(call) do { \
	VkResult _r = (call); \
	if (_r != VK_SUCCESS) { \
		fprintf(stderr, "Vulkan error %d at %s:%d\n", _r, __FILE__, __LINE__); \
		exit(1); \
	} \
} while (0)

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits,
				 VkMemoryPropertyFlags props)
{
	VkPhysicalDeviceMemoryProperties mem_props;
	uint32_t i;

	vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
	for (i = 0; i < mem_props.memoryTypeCount; i++) {
		if ((type_bits & (1 << i)) &&
		    (mem_props.memoryTypes[i].propertyFlags & props) == props)
			return i;
	}
	fprintf(stderr, "Failed to find suitable memory type\n");
	exit(1);
}

static void vk_init(struct vk_state *vk, int use_priority)
{
	/* Instance */
	VkApplicationInfo app_info = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "vkload",
		.apiVersion = VK_API_VERSION_1_1,
	};
	VkInstanceCreateInfo inst_info = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app_info,
	};
	VK_CHECK(vkCreateInstance(&inst_info, NULL, &vk->instance));

	/* Physical device */
	uint32_t dev_count = 0;
	vkEnumeratePhysicalDevices(vk->instance, &dev_count, NULL);
	if (dev_count == 0) {
		fprintf(stderr, "No Vulkan devices found\n");
		exit(1);
	}
	VkPhysicalDevice *devs = calloc(dev_count, sizeof(*devs));
	vkEnumeratePhysicalDevices(vk->instance, &dev_count, devs);
	if (cfg.device_idx >= dev_count) {
		fprintf(stderr, "Device index %u out of range (have %u)\n",
			cfg.device_idx, dev_count);
		exit(1);
	}
	vk->phys_dev = devs[cfg.device_idx];
	free(devs);

	VkPhysicalDeviceProperties dev_props;
	vkGetPhysicalDeviceProperties(vk->phys_dev, &dev_props);
	vk->timestamp_period = dev_props.limits.timestampPeriod;
	fprintf(stderr, "Using device: %s (timestamp period: %.1f ns)\n",
		dev_props.deviceName, vk->timestamp_period);

	/* Queue family — find compute */
	uint32_t qf_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(vk->phys_dev, &qf_count, NULL);
	VkQueueFamilyProperties *qf_props = calloc(qf_count, sizeof(*qf_props));
	vkGetPhysicalDeviceQueueFamilyProperties(vk->phys_dev, &qf_count, qf_props);

	vk->queue_family = UINT32_MAX;
	for (uint32_t i = 0; i < qf_count; i++) {
		if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			vk->queue_family = i;
			break;
		}
	}
	free(qf_props);
	if (vk->queue_family == UINT32_MAX) {
		fprintf(stderr, "No compute queue found\n");
		exit(1);
	}

	/* Logical device */
	float priority = 1.0f;
	VkQueueGlobalPriorityKHR prio = use_priority ? cfg.global_priority : 0;
	VkDeviceQueueGlobalPriorityCreateInfoKHR gp_ci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
		.globalPriority = prio,
	};
	VkDeviceQueueCreateInfo queue_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.pNext = prio ? &gp_ci : NULL,
		.queueFamilyIndex = vk->queue_family,
		.queueCount = 1,
		.pQueuePriorities = &priority,
	};
	const char *dev_exts[] = { "VK_KHR_global_priority" };
	VkDeviceCreateInfo dev_ci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &queue_info,
		.enabledExtensionCount = prio ? 1 : 0,
		.ppEnabledExtensionNames = dev_exts,
	};
	if (prio)
		fprintf(stderr, "Queue priority: %s\n",
			priority_name(prio));
	VK_CHECK(vkCreateDevice(vk->phys_dev, &dev_ci, NULL, &vk->device));
	vkGetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);

	/* Command pool */
	VkCommandPoolCreateInfo pool_ci = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = vk->queue_family,
	};
	VK_CHECK(vkCreateCommandPool(vk->device, &pool_ci, NULL, &vk->cmd_pool));

	/* Buffer — 256KB, device-local + host-visible */
	uint32_t buf_size = 256 * 1024;
	VkBufferCreateInfo buf_ci = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = buf_size,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VK_CHECK(vkCreateBuffer(vk->device, &buf_ci, NULL, &vk->buffer));

	VkMemoryRequirements mem_req;
	vkGetBufferMemoryRequirements(vk->device, vk->buffer, &mem_req);
	VkMemoryAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_req.size,
		.memoryTypeIndex = find_memory_type(vk->phys_dev,
			mem_req.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
	};
	VK_CHECK(vkAllocateMemory(vk->device, &alloc_info, NULL, &vk->buffer_mem));
	VK_CHECK(vkBindBufferMemory(vk->device, vk->buffer, vk->buffer_mem, 0));

	/* Descriptor set layout + pool */
	VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
	};
	VkDescriptorSetLayoutCreateInfo dsl_ci = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};
	VK_CHECK(vkCreateDescriptorSetLayout(vk->device, &dsl_ci, NULL, &vk->desc_layout));

	VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
	};
	VkDescriptorPoolCreateInfo dp_ci = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};
	VK_CHECK(vkCreateDescriptorPool(vk->device, &dp_ci, NULL, &vk->desc_pool));

	VkDescriptorSetAllocateInfo ds_ai = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk->desc_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk->desc_layout,
	};
	VK_CHECK(vkAllocateDescriptorSets(vk->device, &ds_ai, &vk->desc_set));

	VkDescriptorBufferInfo buf_info = {
		.buffer = vk->buffer,
		.offset = 0,
		.range = buf_size,
	};
	VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vk->desc_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &buf_info,
	};
	vkUpdateDescriptorSets(vk->device, 1, &write, 0, NULL);

	/* Pipeline layout with push constant */
	VkPushConstantRange pc_range = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = sizeof(uint32_t),
	};
	VkPipelineLayoutCreateInfo pl_ci = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &vk->desc_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pc_range,
	};
	VK_CHECK(vkCreatePipelineLayout(vk->device, &pl_ci, NULL, &vk->pipe_layout));

	/* Shader module */
	VkShaderModuleCreateInfo sm_ci = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(shader_spirv),
		.pCode = shader_spirv,
	};
	VkShaderModule shader;
	VK_CHECK(vkCreateShaderModule(vk->device, &sm_ci, NULL, &shader));

	/* Compute pipeline */
	VkComputePipelineCreateInfo cp_ci = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = shader,
			.pName = "main",
		},
		.layout = vk->pipe_layout,
	};
	VK_CHECK(vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &cp_ci,
					  NULL, &vk->pipeline));
	vkDestroyShaderModule(vk->device, shader, NULL);

	/* Timestamp query pool */
	VkQueryPoolCreateInfo qp_ci = {
		.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
		.queryType = VK_QUERY_TYPE_TIMESTAMP,
		.queryCount = 2, /* begin + end per job */
	};
	VK_CHECK(vkCreateQueryPool(vk->device, &qp_ci, NULL, &vk->query_pool));
}

static void vk_cleanup(struct vk_state *vk)
{
	vkDeviceWaitIdle(vk->device);
	vkDestroyQueryPool(vk->device, vk->query_pool, NULL);
	vkDestroyPipeline(vk->device, vk->pipeline, NULL);
	vkDestroyPipelineLayout(vk->device, vk->pipe_layout, NULL);
	vkDestroyDescriptorPool(vk->device, vk->desc_pool, NULL);
	vkDestroyDescriptorSetLayout(vk->device, vk->desc_layout, NULL);
	vkFreeMemory(vk->device, vk->buffer_mem, NULL);
	vkDestroyBuffer(vk->device, vk->buffer, NULL);
	vkDestroyCommandPool(vk->device, vk->cmd_pool, NULL);
	vkDestroyDevice(vk->device, NULL);
	vkDestroyInstance(vk->instance, NULL);
}

/* ======================== Job submission ======================== */

static VkCommandBuffer alloc_cmd(struct vk_state *vk)
{
	VkCommandBuffer cmd;
	VkCommandBufferAllocateInfo ai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk->cmd_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VK_CHECK(vkAllocateCommandBuffers(vk->device, &ai, &cmd));
	return cmd;
}

static void submit_job(struct vk_state *vk, VkCommandBuffer cmd, VkFence fence)
{
	VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
	};
	VK_CHECK(vkResetFences(vk->device, 1, &fence));
	VK_CHECK(vkQueueSubmit(vk->queue, 1, &submit, fence));
}

/* ======================== Benchmark resources ======================== */

struct bench_resources {
	VkCommandBuffer *cmds;
	VkFence *fences;
	uint64_t *submit_times;
	uint32_t count;
};

static void bench_resources_init(struct vk_state *vk, struct bench_resources *res,
				 uint32_t count, int signaled)
{
	res->count = count;
	res->cmds = calloc(count, sizeof(*res->cmds));
	res->fences = calloc(count, sizeof(*res->fences));
	res->submit_times = calloc(count, sizeof(*res->submit_times));

	for (uint32_t i = 0; i < count; i++) {
		res->cmds[i] = alloc_cmd(vk);
		VkFenceCreateInfo fence_ci = {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0,
		};
		VK_CHECK(vkCreateFence(vk->device, &fence_ci, NULL, &res->fences[i]));
	}
}

static void bench_resources_cleanup(struct vk_state *vk, struct bench_resources *res)
{
	for (uint32_t i = 0; i < res->count; i++) {
		vkDestroyFence(vk->device, res->fences[i], NULL);
		vkFreeCommandBuffers(vk->device, vk->cmd_pool, 1, &res->cmds[i]);
	}
	free(res->cmds);
	free(res->fences);
	free(res->submit_times);
}

static void record_job(struct vk_state *vk, VkCommandBuffer cmd)
{
	VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

	vkCmdResetQueryPool(cmd, vk->query_pool, 0, 2);
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			    vk->query_pool, 0);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				vk->pipe_layout, 0, 1, &vk->desc_set, 0, NULL);
	vkCmdPushConstants(cmd, vk->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
			   0, sizeof(uint32_t), &vk->loop_count);
	/* Dispatch 1024 invocations (16 workgroups of 64 threads). */
	vkCmdDispatch(cmd, 16, 1, 1);

	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			    vk->query_pool, 1);

	VK_CHECK(vkEndCommandBuffer(cmd));
}

static uint64_t submit_and_wait(struct vk_state *vk, VkCommandBuffer cmd,
				VkFence fence)
{
	submit_job(vk, cmd, fence);
	VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));

	uint64_t timestamps[2];
	VK_CHECK(vkGetQueryPoolResults(vk->device, vk->query_pool, 0, 2,
				       sizeof(timestamps), timestamps,
				       sizeof(uint64_t),
				       VK_QUERY_RESULT_64_BIT |
				       VK_QUERY_RESULT_WAIT_BIT));

	return (uint64_t)((timestamps[1] - timestamps[0]) * vk->timestamp_period);
}

/* ======================== Calibration ======================== */

static void calibrate(struct vk_state *vk, uint32_t target_us)
{
	VkCommandBuffer cmd = alloc_cmd(vk);
	VkFenceCreateInfo fence_ci = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
	};
	VkFence fence;
	VK_CHECK(vkCreateFence(vk->device, &fence_ci, NULL, &fence));

	uint32_t lo = 100, hi = 100000000;
	uint64_t target_ns = (uint64_t)target_us * 1000;

	/* Binary search for loop count that gives target duration */
	for (int iter = 0; iter < 30; iter++) {
		vk->loop_count = lo + (hi - lo) / 2;
		VK_CHECK(vkResetCommandBuffer(cmd, 0));
		record_job(vk, cmd);
		uint64_t gpu_ns = submit_and_wait(vk, cmd, fence);

		if (gpu_ns < target_ns * 90 / 100)
			lo = vk->loop_count;
		else if (gpu_ns > target_ns * 110 / 100)
			hi = vk->loop_count;
		else
			break; /* within 10% */

		if (hi - lo <= 1)
			break;
	}

	/* Final measurement */
	VK_CHECK(vkResetCommandBuffer(cmd, 0));
	record_job(vk, cmd);
	uint64_t final_ns = submit_and_wait(vk, cmd, fence);

	fprintf(stderr, "Calibrated: loop_count=%u, GPU duration=%.1f us (target=%u us)\n",
		vk->loop_count, final_ns / 1000.0, target_us);

	vkDestroyFence(vk->device, fence, NULL);
	vkFreeCommandBuffers(vk->device, vk->cmd_pool, 1, &cmd);
}

/* ======================== Ring-buffer benchmark loop ======================== */

static void run_benchmark(struct vk_state *vk, FILE *out)
{
	uint32_t depth = cfg.queue_depth;
	struct bench_resources res;
	uint32_t slot = 0;
	uint32_t job_id = 0;

	bench_resources_init(vk, &res, depth, 1);

	fprintf(out, "job_id,submit_ns,complete_ns,elapsed_ns\n");

	uint64_t start = now_ns();
	uint64_t end = start + (uint64_t)cfg.duration_ms * 1000000;
	uint64_t next_submit = start;
	uint64_t interval_ns = cfg.rate_hz ? 1000000000ULL / cfg.rate_hz : 0;

	while (now_ns() < end) {
		if (interval_ns && now_ns() < next_submit)
			sleep_until_ns(next_submit);

		VK_CHECK(vkWaitForFences(vk->device, 1, &res.fences[slot],
					 VK_TRUE, UINT64_MAX));

		/* Harvest completed job BEFORE overwriting the slot */
		if (job_id >= depth) {
			uint64_t t_complete = now_ns();
			fprintf(out, "%u,%lu,%lu,%lu\n",
				job_id - depth,
				res.submit_times[slot],
				t_complete,
				t_complete - res.submit_times[slot]);
		}

		VK_CHECK(vkResetCommandBuffer(res.cmds[slot], 0));
		record_job(vk, res.cmds[slot]);
		res.submit_times[slot] = now_ns();
		submit_job(vk, res.cmds[slot], res.fences[slot]);

		if (interval_ns)
			next_submit += interval_ns;

		slot = (slot + 1) % depth;
		job_id++;
	}

	/* Drain remaining in-flight jobs */
	vkDeviceWaitIdle(vk->device);
	uint64_t t_drain = now_ns();
	uint32_t in_flight = (job_id < depth) ? job_id : depth;
	for (uint32_t i = 0; i < in_flight; i++) {
		uint32_t s = (slot + i) % depth;
		uint32_t drain_id = job_id - in_flight + i;

		fprintf(out, "%u,%lu,%lu,%lu\n",
			drain_id, res.submit_times[s], t_drain,
			t_drain - res.submit_times[s]);
	}

	bench_resources_cleanup(vk, &res);

	fprintf(stderr, "Completed %u jobs in %lu ms\n",
		job_id, (now_ns() - start) / 1000000);
}

/* ======================== CLI ======================== */

static void usage(const char *prog)
{
	fprintf(stderr,
		"Usage: %s [OPTIONS]\n"
		"  --device <idx>       Vulkan device index (default: 0)\n"
		"  --job-us <us>        Target GPU job duration (default: 1000)\n"
		"  --rate <hz>          Submission rate, 0=continuous (default: 0)\n"
		"  --duration <ms>      Benchmark duration (default: 10000)\n"
		"  --queue-depth <n>    Max in-flight jobs (default: 4)\n"
		"  --output <file>      CSV output (default: stdout)\n"
		"  --priority <p>       Queue priority: low, medium, high, realtime\n"
		"  --sync-dir <dir>     Barrier directory for multi-process sync\n"
		"  --sync-count <n>     Number of processes to wait for at barrier\n"
		"  --calibrate-only     Calibrate and exit\n"
		"  --help               Show this help\n",
		prog);
}

int main(int argc, char **argv)
{
	static struct option long_opts[] = {
		{"device",         required_argument, NULL, 'd'},
		{"job-us",         required_argument, NULL, 'j'},
		{"rate",           required_argument, NULL, 'r'},
		{"duration",       required_argument, NULL, 'D'},
		{"queue-depth",    required_argument, NULL, 'q'},
		{"output",         required_argument, NULL, 'o'},
		{"priority",       required_argument, NULL, 'p'},
		{"sync-dir",       required_argument, NULL, 's'},
		{"sync-count",     required_argument, NULL, 'n'},
		{"calibrate-only", no_argument,       NULL, 'c'},
		{"help",           no_argument,       NULL, 'h'},
		{0, 0, 0, 0}
	};

	int opt;
	while ((opt = getopt_long(argc, argv, "d:j:r:D:q:b:o:p:s:n:ch", long_opts, NULL)) != -1) {
		switch (opt) {
		case 'd': cfg.device_idx = atoi(optarg); break;
		case 'j': cfg.job_us = atoi(optarg); break;
		case 'r': cfg.rate_hz = atoi(optarg); break;
		case 'D': cfg.duration_ms = atoi(optarg); break;
		case 'q': cfg.queue_depth = atoi(optarg); break;
		case 'o': cfg.output = optarg; break;
		case 'p': cfg.global_priority = parse_priority(optarg); break;
		case 's': cfg.sync_dir = optarg; break;
		case 'n': cfg.sync_count = atoi(optarg); break;
		case 'c': cfg.calibrate_only = 1; break;
		case 'h': usage(argv[0]); return 0;
		default:  usage(argv[0]); return 1;
		}
	}

	struct vk_state vk = {0};

	/*
	 * Calibrate on a device without priority.
	 *
	 * Then recreate the device with the target priority for the
	 * actual benchmark. This also gives a fresh drm_sched_entity
	 * without accumulated runtime debt from calibration.
	 */
	vk_init(&vk, 0);
	calibrate(&vk, cfg.job_us);

	if (cfg.calibrate_only) {
		vk_cleanup(&vk);
		return 0;
	}

	uint32_t saved_loop_count = vk.loop_count;
	vk_cleanup(&vk);
	memset(&vk, 0, sizeof(vk));
	vk_init(&vk, 1);
	vk.loop_count = saved_loop_count;

	sync_barrier();

	FILE *out = stdout;
	if (cfg.output) {
		out = fopen(cfg.output, "w");
		if (!out) {
			perror("fopen");
			vk_cleanup(&vk);
			return 1;
		}
	}

	run_benchmark(&vk, out);

	if (cfg.output)
		fclose(out);
	vk_cleanup(&vk);
	return 0;
}
