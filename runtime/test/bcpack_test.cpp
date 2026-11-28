// Independent synthetic goldens; no GDK or game files needed to run this test.
#include "../src/bcpack.h"
#include "../../third_party/x2w-bcpack/bcpack_decoder.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<uint8_t> unhex(const std::string& text) {
    std::vector<uint8_t> result;
    for (size_t i = 0; i < text.size(); i += 2)
        result.push_back(static_cast<uint8_t>(std::stoul(text.substr(i, 2), nullptr, 16)));
    return result;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream input(argv[1]);
    if (!input) return 2;
    unsigned cases = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        unsigned codec;
        std::string name, packed, expected;
        std::istringstream row(line);
        if (!(row >> codec >> name >> packed >> expected)) return 2;
        auto source = unhex(packed), want = unhex(expected);
        std::vector<uint8_t> output(want.size() + 32, 0xa5);
        const auto target = std::span(output).subspan(16, want.size());
        for (int path = 0; path < x2w_bcpack::LANE_PATH_COUNT; ++path) {
            if (!x2w_bcpack::lane_path_supported(path)) continue;
            x2w_bcpack::lane_path_override = path;
            std::fill(output.begin(), output.end(), 0xa5);
            if (leonida::bcpack::decode(codec, source, target) != leonida::bcpack::Quality::confirmed ||
                !std::equal(want.begin(), want.end(), target.begin()) ||
                !std::all_of(output.begin(), output.begin() + 16, [](auto v) { return v == 0xa5; }) ||
                !std::all_of(output.end() - 16, output.end(), [](auto v) { return v == 0xa5; })) {
                std::cerr << "golden failed: " << name << " path " << path << '\n';
                return 1;
            }
        }
        x2w_bcpack::lane_path_override = -1;
        if (name.ends_with("-request")) {
            for (size_t cut : {size_t(0), size_t(1), source.size() / 2}) {
                std::fill(output.begin(), output.end(), 0xa5);
                if (leonida::bcpack::decode(codec, std::span(source).first(cut), target) !=
                    leonida::bcpack::Quality::failed ||
                    !std::all_of(output.begin(), output.end(), [](auto v) { return v == 0xa5; })) return 1;
            }
            source.push_back(0xff);
            if (leonida::bcpack::decode(codec, source, target) != leonida::bcpack::Quality::failed) return 1;
            if (leonida::bcpack::decode(codec, source, target.first(target.size() - 1)) !=
                leonida::bcpack::Quality::failed) return 1;
        }
        ++cases;
    }
    // Exhaust every non-reserved rANS slot of all 128 models. Check the fast
    // reciprocal inverse against each independently computed forward interval.
    for (uint8_t ab = 0; ab < 8; ++ab) for (uint8_t q = 0; q < 16; ++q) {
        for (uint16_t symbol = 0; symbol < (1u << (ab + 1)); ++symbol) {
            uint32_t f, start;
            x2w_bcpack::model_forward(ab, q, symbol, f, start);
            for (uint32_t slot = start; slot < start + f; ++slot)
                if (x2w_bcpack::model_inverse_fast(ab, q, static_cast<uint16_t>(slot)) != symbol) return 1;
        }
    }
    std::cout << "BCPack: " << cases << " independent goldens on every supported CPU path; all 128 models; malformed input guards passed\n";
    return cases ? 0 : 1;
}
