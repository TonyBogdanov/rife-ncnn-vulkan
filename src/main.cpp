#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <clocale>

#include "cpu.h"
#include "gpu.h"
#include "platform.h"

#include "rife.h"
#include "filesystem_utils.h"

#ifdef _WIN32
    #include <io.h>
    #include "wic_image.h"
#endif

constexpr uint8_t SIGNAL_WORK = 0;
constexpr uint8_t SIGNAL_EXIT = 1;

void *readBytes(const size_t size) {
    auto *buffer = static_cast<unsigned char *>(malloc(size));
    const auto read = fread(buffer, size, 1, stdin);

    if (1 != read) {
        fprintf(stderr, "Error: Failed to read %lu bytes from stdin.", size);
        exit(EXIT_FAILURE);
    }

    return buffer;
}

uint8_t readUint8() {
    uint8_t value;
    const size_t read = fread(&value, sizeof(uint8_t), 1, stdin);

    if (1 != read) {
        fprintf(stderr, "Error: Failed to read uint8_t from stdin.");
        exit(EXIT_FAILURE);
    }

    return value;
}

uint64_t readUint64() {
    uint64_t value;
    const auto read = fread(&value, sizeof(uint64_t), 1, stdin);

    if (1 != read) {
        fprintf(stderr, "Error: Failed to read uint64_t from stdin.");
        exit(EXIT_FAILURE);
    }

    return (value & 0xFF) << 56 |
           (value & 0xFF00) << 40 |
           (value & 0xFF0000) << 24 |
           (value & 0xFF000000) << 8 |
           value >> 8 & 0xFF000000 |
           value >> 24 & 0xFF0000 |
           value >> 40 & 0xFF00 |
           value >> 56 & 0xFF;
}

void writeBytes(const void *bytes, const size_t size) {
    const auto write = fwrite(bytes, sizeof(unsigned char), size, stdout);
    fflush(stdout);

    if (size != write) {
        fprintf(stderr, "Error: Failed to write %lu bytes to stdout, wrote %lu instead.", size, write);
    }
}

void writeUint8(const uint8_t value) {
    writeBytes(&value, sizeof(uint8_t));
}

void writeUint64(const uint64_t value) {
    writeBytes(&value, sizeof(uint64_t));
}

#ifdef _WIN32
int wmain() {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
#elif __linux__
int main() {
#endif

    setlocale(LC_ALL, "");
    ncnn::create_gpu_instance();

    const path_t modelDir = sanitize_dirpath(PATHSTR("./model"));
    int gpuID = ncnn::get_default_gpu_index();

    const auto width = readUint64();
    const auto height = readUint64();

    RIFE *rife = new RIFE(gpuID, false, false, 1920 < width || 1920 < height, 1, false, true);
    rife->load(modelDir);

    writeUint8(0); // ready

    while (true) {
        const auto command = readUint8();
        if (SIGNAL_EXIT == command) {
            break;
        }

        if (SIGNAL_WORK != command) {
            fprintf(stderr, "Error: Invalid signal. Terminating...\n");
            exit(EXIT_FAILURE);
        }

        ncnn::Mat in0 = ncnn::Mat(width, height, readBytes(width * height * 3), 3, 3);
        ncnn::Mat in1 = ncnn::Mat(width, height, readBytes(width * height * 3), 3, 3);

        ncnn::Mat out = ncnn::Mat(in0.w, in0.h, 3, 3);
        rife->process(in0, in1, 0.5f, out);

        writeBytes(out.data, width * height * 3);
        free(in0.data);
        free(in1.data);
    }

    ncnn::destroy_gpu_instance();
    return 0;
}
