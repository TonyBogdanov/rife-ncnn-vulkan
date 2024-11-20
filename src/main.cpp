#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#include <locale.h>
#include <stdint.h>

#include "wic_image.h"

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"

#include "rife.h"
#include "filesystem_utils.h"

static uint16_t frames, width, height;

static uint64_t readUint64() {
	unsigned char* data = (unsigned char*)malloc(8);
	size_t read = fread(data, 1, 8, stdin);

	if (8 != read) {
		fprintf(stderr, "Error: Failed to read uint64, expected: %d bytes, read: %d instead.\n", 8, read);
		exit(1);
	}

	uint64_t value = (static_cast<uint64_t>(data[0]) << 56) |
		(static_cast<uint64_t>(data[1]) << 48) |
		(static_cast<uint64_t>(data[2]) << 40) |
		(static_cast<uint64_t>(data[3]) << 32) |
		(static_cast<uint64_t>(data[4]) << 24) |
		(static_cast<uint64_t>(data[5]) << 16) |
		(static_cast<uint64_t>(data[6]) << 8) |
		static_cast<uint64_t>(data[7]);

	return value;
}

static void decodeImage(ncnn::Mat& image)
{
	size_t size = (size_t)width * height * 3;
	unsigned char* pixeldata = (unsigned char*)malloc(size);

	size_t read = fread(pixeldata, 1, size, stdin);
	if (read != width * height * 3) {
		fprintf(stderr, "Error: Failed to read frame, expected: %d bytes, read: %d instead.\n", size, read);
		return;
	}

	image = ncnn::Mat(width, height, (void*)pixeldata, (size_t)3, 3);
}

static void encodeImage(const ncnn::Mat& image)
{
	fwrite(image.data, 1, image.w * image.h * 3, stdout);
	fflush(stdout);
}

int wmain()
{
	_setmode(_fileno(stdin), _O_BINARY);
	_setmode(_fileno(stdout), _O_BINARY);

	setlocale(LC_ALL, "");

	frames = readUint64();
	width = readUint64();
	height = readUint64();

	fprintf(stderr, "Interpolating %d [%dx%d] frames.\n", frames, width, height);

	CoInitializeEx(NULL, COINIT_MULTITHREADED);
	ncnn::create_gpu_instance();

	path_t modeldir = sanitize_dirpath(PATHSTR("rife-ncnn-vulkan/rife-v4"));
	int gpuid = ncnn::get_default_gpu_index();

	int cpu_count = std::max(1, ncnn::get_cpu_count());
	int gpu_count = ncnn::get_gpu_count();

	RIFE* rife = new RIFE(gpuid, 0, 0, 0, 1, false, true);
	rife->load(modeldir);

	for ( int i = 0; i < frames - 1; i++ ) {
		ncnn::Mat in0;
		ncnn::Mat in1;
		ncnn::Mat out;

		decodeImage(in0);
		decodeImage(in1);

		out = ncnn::Mat(in0.w, in0.h, (size_t)3, 3);
		rife->process(in0, in1, 0.5f, out);

		encodeImage(out);

		{
			unsigned char* pixeldata = (unsigned char*)in0.data;
			free(pixeldata);
		}

		{
			unsigned char* pixeldata = (unsigned char*)in1.data;
			free(pixeldata);
		}
	}

	ncnn::destroy_gpu_instance();
	return 0;
}
