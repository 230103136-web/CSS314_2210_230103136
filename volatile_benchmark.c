
#include <stdint.h>

void volatile_increment(
    volatile uint64_t *data,
    int index,
    uint64_t iterations
) {
    volatile uint64_t *slot = &data[index];

    for (uint64_t i = 0; i < iterations; i++) {
        uint64_t value = *slot;
        *slot = value + 1;
    }
}
