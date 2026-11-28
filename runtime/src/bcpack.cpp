#include "bcpack.h"
#include "../../third_party/x2w-bcpack/bcpack_decoder.h"

namespace leonida::bcpack {
Quality decode(unsigned codec, std::span<const std::uint8_t> source,
               std::span<std::uint8_t> output, const char** failure) {
    if (failure) *failure = "bcpack-invalid-stream";
    size_t produced = 0;
    if (!x2w_bcpack::decode_bcpack(source.data(), source.size(), codec,
                                 output.data(), output.size(), &produced) || produced != output.size())
        return Quality::failed;
    if (failure) *failure = nullptr;
    return Quality::confirmed;
}
} // namespace leonida::bcpack
