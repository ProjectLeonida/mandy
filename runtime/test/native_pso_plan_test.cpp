// Portable unit tests for native_pso::HashMap, record_digests, decode_state_b, and plan (src/native_pso.h).
// Synthetic descriptions and digests only; no game data, no D3D12 or Vulkan.
#include "../src/native_pso.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace np = leonida::native_pso;

#define CHECK(cond) do {                                             \
    if (!(cond)) {                                                   \
        std::fprintf(stderr, "failed at line %d: %s\n", __LINE__, #cond); \
        return 1;                                                    \
    }                                                                \
} while (false)

namespace {

np::Digest make_digest(std::uint64_t a, std::uint64_t b) {
    np::Digest d{};
    std::memcpy(d.data(), &a, 8);
    std::memcpy(d.data() + 8, &b, 8);
    return d;
}

void append_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    const std::size_t at = out.size();
    out.resize(at + 4);
    std::memcpy(out.data() + at, &v, 4);
}

void append_record(std::vector<std::uint8_t>& out, const np::Digest& d) {
    append_u32(out, 0x43);
    out.insert(out.end(), d.begin(), d.end());
}

std::vector<std::uint8_t> make_state_b(std::uint8_t rt_count = 1, std::uint8_t topo = 3) {
    std::vector<std::uint8_t> b(np::kStateBHeader, 0);
    b[0x25] = rt_count;
    b[0x26] = topo;
    b[0x28] = 1;
    return b;
}

}  // namespace

int main() {
    // 1. hex() and parse_digest() round-trip and validation.
    {
        const np::Digest d = make_digest(0x0123456789abcdefull, 0xfedcba9876543210ull);
        const std::string h = np::hex(d);
        CHECK(h.size() == 32);
        const auto parsed = np::parse_digest(h);
        CHECK(parsed && *parsed == d);
        CHECK(!np::parse_digest(""));
        CHECK(!np::parse_digest(h.substr(0, 31)));
        CHECK(!np::parse_digest(h + "0"));
        std::string upper = h;
        upper[0] = 'A';
        CHECK(!np::parse_digest(upper));
    }

    // 2. record_digests(): empty, duplicate dedup preserving order, unaligned 0x43 ignored, and >16 distinct spill.
    {
        std::vector<std::uint8_t> rec;
        const np::Digest d1 = make_digest(1, 100);
        const np::Digest d2 = make_digest(2, 200);
        append_u32(rec, 0x11);
        append_record(rec, d1);
        append_u32(rec, 0x4300);  // misaligned 0x43 byte inside another word
        append_record(rec, d1);   // duplicate must be deduplicated
        append_record(rec, d2);
        append_u32(rec, 0x43);    // trailing 0x43 with <16 bytes after it must be ignored
        append_u32(rec, 0x99);
        const auto found = np::record_digests(rec);
        CHECK(found.size() == 2 && found[0] == d1 && found[1] == d2);

        // Exercise the >16 distinct digest hash-set spill path + duplicates after spill.
        std::vector<std::uint8_t> big;
        for (std::uint64_t i = 0; i < 32; ++i) append_record(big, make_digest(0x1000, i));
        for (std::uint64_t i = 0; i < 32; ++i) append_record(big, make_digest(0x1000, i));  // all duplicates
        const auto big_found = np::record_digests(big);
        CHECK(big_found.size() == 32);
        for (std::uint64_t i = 0; i < 32; ++i) CHECK(big_found[i] == make_digest(0x1000, i));
    }

    // 3. HashMap::parse() validation, CRLF, comments, identical duplicate allowed, conflicting duplicate rejected.
    {
        np::HashMap map;
        std::string err;
        const np::Digest vs_d = make_digest(0xaa, 1);
        const np::Digest ps_d = make_digest(0xbb, 2);
        const np::Digest gs_d = make_digest(0xcc, 3);
        const np::Digest ms_d = make_digest(0xdd, 4);
        const np::Digest cs_d = make_digest(0xee, 5);
        std::string catalog = std::string(np::HashMap::kHeader) + "\r\n"
            "# comment line\r\n\r\n" +
            np::hex(vs_d) + "\tvs\tlibrary/vs0.dxbc\r\n" +
            np::hex(vs_d) + "\tvs\tlibrary/vs0.dxbc\r\n" +  // identical duplicate allowed
            np::hex(ps_d) + "\tps\tlibrary/ps0.dxbc\n" +
            np::hex(gs_d) + "\tgs\tlibrary/gs0.dxbc\n" +
            np::hex(ms_d) + "\tms\tlibrary/ms0.dxbc\n" +
            np::hex(cs_d) + "\tcs\tlibrary/cs0.dxbc\n";
        CHECK(map.parse(catalog, &err));
        CHECK(map.size() == 5);
        CHECK(map.find(vs_d) && map.find(vs_d)->stage == "vs" && map.find(vs_d)->path == "library/vs0.dxbc");
        CHECK(map.find(ps_d) && map.find(ps_d)->stage == "ps");
        CHECK(!map.find(make_digest(0xff, 99)));

        // Conflicting path or stage for an existing digest must fail and clear map.
        np::HashMap bad;
        CHECK(!bad.parse(catalog + np::hex(vs_d) + "\tvs\tlibrary/other.dxbc\n", &err) && bad.size() == 0);
        CHECK(!bad.parse(catalog + np::hex(vs_d) + "\tps\tlibrary/vs0.dxbc\n", &err) && bad.size() == 0);
        CHECK(!bad.parse("", &err));
        CHECK(!bad.parse("# wrong header\n", &err));
        CHECK(!bad.parse(std::string(np::HashMap::kHeader) + "\nbadline\n", &err));
        CHECK(!bad.parse(std::string(np::HashMap::kHeader) + "\n" + np::hex(vs_d) + "\t\tpath.dxbc\n", &err));

        // 4. plan(): VS+PS, VS+GS+PS, MS+PS, duplicate same digest in one block (not ambiguous),
        //    two distinct digests in one block (ambiguous), missing, unsupported, malformed.
        const auto state_b = make_state_b();
        std::vector<std::uint8_t> rec_vs, rec_ps, rec_gs, rec_ms, rec_cs, rec_ambig, rec_unknown;
        append_record(rec_vs, vs_d);
        append_record(rec_vs, vs_d);  // same digest repeated in one block: NOT ambiguous
        append_record(rec_ps, ps_d);
        append_record(rec_gs, gs_d);
        append_record(rec_ms, ms_d);
        append_record(rec_cs, cs_d);
        append_record(rec_ambig, vs_d);
        append_record(rec_ambig, ps_d);
        append_record(rec_unknown, make_digest(0x1234, 0x5678));
        const std::uint8_t dummy_code[4] = {1, 2, 3, 4};

        // VS + PS ok
        np::GraphicsDesc desc{};
        desc.state_b = {state_b.data(), state_b.size()};
        desc.stages[0] = {{dummy_code, sizeof(dummy_code)}, {rec_vs.data(), rec_vs.size()}};
        desc.stages[1] = {{dummy_code, sizeof(dummy_code)}, {rec_ps.data(), rec_ps.size()}};
        auto p = np::plan(desc, map);
        CHECK(p.status == np::PlanStatus::ok && !p.mesh && p.stages.size() == 2);
        CHECK(p.stage("vs") == map.find(vs_d) && p.stage("ps") == map.find(ps_d));

        // VS + GS + PS ok
        desc.stages[2] = {{dummy_code, sizeof(dummy_code)}, {rec_gs.data(), rec_gs.size()}};
        p = np::plan(desc, map);
        CHECK(p.status == np::PlanStatus::ok && !p.mesh && p.stages.size() == 3);

        // MS + PS ok
        desc.stages[0] = {{dummy_code, sizeof(dummy_code)}, {rec_ms.data(), rec_ms.size()}};
        desc.stages[2] = {};
        p = np::plan(desc, map);
        CHECK(p.status == np::PlanStatus::ok && p.mesh && p.stage("ms") == map.find(ms_d));

        // MS + GS unsupported
        desc.stages[2] = {{dummy_code, sizeof(dummy_code)}, {rec_gs.data(), rec_gs.size()}};
        p = np::plan(desc, map);
        CHECK(p.status == np::PlanStatus::unsupported);

        // CS stage unsupported
        desc.stages[0] = {{dummy_code, sizeof(dummy_code)}, {rec_cs.data(), rec_cs.size()}};
        desc.stages[2] = {};
        p = np::plan(desc, map);
        CHECK(p.status == np::PlanStatus::unsupported && p.failed_block == 0);

        // Ambiguous two distinct digests in one stage block
        desc.stages[0] = {{dummy_code, sizeof(dummy_code)}, {rec_ambig.data(), rec_ambig.size()}};
        p = np::plan(desc, map);
        CHECK(p.status == np::PlanStatus::ambiguous && p.failed_block == 0);

        // Unknown digest -> missing
        desc.stages[0] = {{dummy_code, sizeof(dummy_code)}, {rec_unknown.data(), rec_unknown.size()}};
        p = np::plan(desc, map);
        CHECK(p.status == np::PlanStatus::missing && p.failed_block == 0);

        // Code without records -> missing
        desc.stages[0] = {{dummy_code, sizeof(dummy_code)}, {}};
        p = np::plan(desc, map);
        CHECK(p.status == np::PlanStatus::missing && p.failed_block == 0);

        // Malformed pair (null data with non-zero size)
        desc.stages[0] = {{nullptr, 16}, {rec_vs.data(), rec_vs.size()}};
        p = np::plan(desc, map);
        CHECK(p.status == np::PlanStatus::malformed);
    }

    // 5. decode_state_b() input elements and bounds checks.
    {
        auto b = make_state_b(2, 3);
        const std::uint32_t elem_count = 1;
        std::memcpy(b.data() + 0x4c, &elem_count, 4);
        const char sem[] = "POSITION";
        append_u32(b, static_cast<std::uint32_t>(sizeof(sem)));  // includes NUL
        b.insert(b.end(), sem, sem + sizeof(sem));
        for (std::uint32_t v : {0u, 6u, 0u, 12u, 0u, 0u}) append_u32(b, v);
        const char* why = nullptr;
        const auto s = np::decode_state_b(b, &why);
        CHECK(s && s->rt_count == 2 && s->elements.size() == 1 && s->elements[0].name == "POSITION" &&
              s->elements[0].format == 6 && s->elements[0].aligned_byte_offset == 12);

        // Truncated element must fail cleanly.
        b.pop_back();
        CHECK(!np::decode_state_b(b, &why));
    }

    std::puts("native_pso_plan: digest hash, record_digests, HashMap, decode_state_b and plan OK");
    return 0;
}
