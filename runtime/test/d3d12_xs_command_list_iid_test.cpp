// Tests console command list interface ID mapping and classification
// (runtime/shims/d3d12_xs_command_list_iid.h).
// Synthetic GUIDs only; no game files or GPU required.
#include <windows.h>
#include <d3d12.h>

#include <cstdio>

#include "d3d12_xs_command_list_iid.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL %s\n", what);
    }
}

void test_console_iid_mapping() {
    using namespace rt::d3d12_xs;

    // Measured console command list IID from retail executable (docs/xdxr.md §6 / TomBox)
    const GUID console_gcl6 = {0xc3827890, 0x4024, 0x474c, {0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74}};
    const GUID expected_pc_gcl6 = {0xc3827890, 0xe548, 0x4cfa, {0x96, 0xcf, 0x56, 0x89, 0xa9, 0x37, 0x0f, 0x80}};

    // Verify PC GUID matches __uuidof(ID3D12GraphicsCommandList6)
    check(IsEqualGUID(expected_pc_gcl6, __uuidof(ID3D12GraphicsCommandList6)),
          "expected_pc_gcl6 matches __uuidof(ID3D12GraphicsCommandList6)");

    // Mapping with mapped flag pointer
    bool mapped = false;
    const GUID& result = command_list_iid_for_pc(console_gcl6, &mapped);
    check(mapped, "console command list IID reports mapped == true");
    check(IsEqualGUID(result, expected_pc_gcl6), "console command list IID maps to PC ID3D12GraphicsCommandList6");
    check(result.Data1 == console_gcl6.Data1, "mapped PC IID preserves identical first 32 bits (Data1)");

    // Mapping with null mapped pointer
    const GUID& result_null = command_list_iid_for_pc(console_gcl6, nullptr);
    check(IsEqualGUID(result_null, expected_pc_gcl6), "command_list_iid_for_pc handles null mapped pointer safely");
}

void test_is_pc_command_list_iid() {
    using namespace rt::d3d12_xs;

    // Every PC command list interface from base through GCL7 must be recognized
    check(is_pc_command_list_iid(__uuidof(ID3D12CommandList)), "ID3D12CommandList recognized as PC IID");
    check(is_pc_command_list_iid(__uuidof(ID3D12GraphicsCommandList)), "ID3D12GraphicsCommandList recognized as PC IID");
    check(is_pc_command_list_iid(__uuidof(ID3D12GraphicsCommandList1)), "ID3D12GraphicsCommandList1 recognized as PC IID");
    check(is_pc_command_list_iid(__uuidof(ID3D12GraphicsCommandList2)), "ID3D12GraphicsCommandList2 recognized as PC IID");
    check(is_pc_command_list_iid(__uuidof(ID3D12GraphicsCommandList3)), "ID3D12GraphicsCommandList3 recognized as PC IID");
    check(is_pc_command_list_iid(__uuidof(ID3D12GraphicsCommandList4)), "ID3D12GraphicsCommandList4 recognized as PC IID");
    check(is_pc_command_list_iid(__uuidof(ID3D12GraphicsCommandList5)), "ID3D12GraphicsCommandList5 recognized as PC IID");
    check(is_pc_command_list_iid(__uuidof(ID3D12GraphicsCommandList6)), "ID3D12GraphicsCommandList6 recognized as PC IID");
    check(is_pc_command_list_iid(__uuidof(ID3D12GraphicsCommandList7)), "ID3D12GraphicsCommandList7 recognized as PC IID");

    // Non-command list interfaces return false
    check(!is_pc_command_list_iid(__uuidof(IUnknown)), "IUnknown is not a PC command list IID");
    check(!is_pc_command_list_iid(__uuidof(ID3D12Device)), "ID3D12Device is not a PC command list IID");
    check(!is_pc_command_list_iid(__uuidof(ID3D12Resource)), "ID3D12Resource is not a PC command list IID");

    // Console command list IID is not a PC IID
    const GUID console_gcl6 = {0xc3827890, 0x4024, 0x474c, {0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74}};
    check(!is_pc_command_list_iid(console_gcl6), "console command list IID is not recognized as a PC IID");
}

void test_pc_and_unknown_iids_unsubstituted() {
    using namespace rt::d3d12_xs;

    // Passing PC IID leaves it unmodified with mapped == false
    bool mapped = true;
    const GUID& pc_gcl6 = command_list_iid_for_pc(__uuidof(ID3D12GraphicsCommandList6), &mapped);
    check(!mapped, "PC GCL6 reports mapped == false");
    check(IsEqualGUID(pc_gcl6, __uuidof(ID3D12GraphicsCommandList6)), "PC GCL6 returned unchanged");

    mapped = true;
    const GUID& pc_gcl1 = command_list_iid_for_pc(__uuidof(ID3D12GraphicsCommandList1), &mapped);
    check(!mapped, "PC GCL1 reports mapped == false");
    check(IsEqualGUID(pc_gcl1, __uuidof(ID3D12GraphicsCommandList1)), "PC GCL1 returned unchanged");

    // Unknown random IID forwarded unchanged with mapped == false
    const GUID unknown = {0x11223344, 0x5566, 0x7788, {0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00}};
    mapped = true;
    const GUID& forwarded = command_list_iid_for_pc(unknown, &mapped);
    check(!mapped, "unknown IID reports mapped == false");
    check(IsEqualGUID(forwarded, unknown), "unknown IID returned unchanged");
}

}  // namespace

int main() {
    test_console_iid_mapping();
    test_is_pc_command_list_iid();
    test_pc_and_unknown_iids_unsubstituted();

    std::printf("%d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures ? 1 : 0;
}
