// dnsapi_offline: the DNS client API (dnsapi.dll) for a mapped title, answered without the network.
//
// Nothing in Leonida goes online. winhttp_offline.cpp owns winhttp.dll and ws2_offline.cpp vets Winsock2, but a title
// that calls dnsapi.dll directly reached the host's DNS client, which sends a query for any name, and dynamic updates,
// server checks and DNS-SD (multicast DNS on the local link) as well (docs/runtime_imports.md listed this as the known
// gap). dnsapi.dll stays host-owned in rt::shims::owns_dll, like ws2_32.dll: a title's LoadLibrary gets the real
// module, and the functions that send nothing (DnsRecordListFree, DnsFree, DnsQueryConfig, the name helpers, the
// cancel functions) are the host's. The ones that can send are vetted here:
//   - DnsQuery_A, DnsQuery_W, DnsQuery_UTF8 and DnsQueryEx answer only "localhost" (any case, with or without the
//     trailing dot) and loopback literals (127.0.0.0/8 in plain decimal, ::1, ::ffff:127.0.0.0/104), the names
//     ws2_offline.cpp lets through. Those go to the host with the network switched off: DNS_QUERY_NO_WIRE_QUERY,
//     NO_NETBT, NO_MULTICAST and NO_HOSTS_FILE are added, the options that ask for the wire (WIRE_ONLY,
//     MULTICAST_ONLY, BYPASS_CACHE, RETURN_MESSAGE) are removed (the host refuses them with the wire off), and server
//     lists are dropped. The host answers these itself (measured on Windows 11: localhost as 127.0.0.1 and ::1, a
//     literal as its own record), and the records are its own, so the title frees them with the host's
//     DnsRecordListFree or DnsFree as usual.
//   - Every other name fails with DNS_ERROR_RECORD_DOES_NOT_EXIST, the host's own answer for a name it cannot resolve
//     without the network (measured with DNS_QUERY_NO_WIRE_QUERY), and *ppQueryResults is NULL. The resolver cache,
//     the hosts file, the machine's own name and other literals are not consulted, as in ws2_offline.cpp, so a title
//     neither reaches the network nor reads what other programs resolved. A leading zero (0127.0.0.1) is refused too.
//   - DnsQueryEx always completes before it returns, also when the request has a completion callback: then the
//     callback is not called, which is how the host answers a name it resolves locally (measured: localhost with a
//     callback returns 0 at once). So no shim query is ever pending, and DnsCancelQuery stays the host's.
//   - DnsQueryRaw, DnsModifyRecordsInSet_A/W/UTF8, DnsReplaceRecordSetA/W/UTF8 (dynamic updates), DnsServiceBrowse,
//     DnsServiceResolve, DnsServiceRegister, DnsServiceDeRegister and DnsStartMulticastQuery (DNS-SD and multicast
//     DNS) fail with ERROR_NETWORK_UNREACHABLE before anything is sent, the Win32 counterpart of ws2_offline's
//     WSAENETUNREACH. They read no argument, and no completion callback is ever called.
//   - DnsValidateServerStatus succeeds and reports DNS_VALSVR_ERROR_UNREACHABLE: the server is never asked.
// Each refused name or function is logged once ("[rt] dnsapi: offline, refused <function> <name>"). Every host
// dnsapi.dll call is resolved at run time from System32, so gta6rt does not link dnsapi.lib. Undocumented exports
// (DnsQueryExA/W/UTF8, DnsValidateServer_A/W, the internal Dns_*/Socket_* helpers) are not covered.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <winsock2.h>
#include <windows.h>
#include <windns.h>

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>

#include "shims.h"

namespace rt::dnsapi {
namespace {

// The options that make the host answer from the machine alone, and the ones it refuses next to them.
constexpr DWORD kNoNetwork = DNS_QUERY_NO_WIRE_QUERY | DNS_QUERY_NO_NETBT | DNS_QUERY_NO_MULTICAST |
                             DNS_QUERY_NO_HOSTS_FILE;
constexpr DWORD kNeedsNetwork = DNS_QUERY_WIRE_ONLY | DNS_QUERY_MULTICAST_ONLY | DNS_QUERY_BYPASS_CACHE |
                                DNS_QUERY_RETURN_MESSAGE;

HMODULE host_dnsapi() {
    static HMODULE mod = LoadLibraryExW(L"dnsapi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    return mod;
}

template <typename Fn>
Fn host_proc(const char* name) {
    HMODULE mod = host_dnsapi();
    return mod ? reinterpret_cast<Fn>(GetProcAddress(mod, name)) : nullptr;
}

// A name as printable ASCII for the log; anything else becomes '?'.
std::string printable(const char* name) {
    std::string out;
    for (; name && *name && out.size() < 255; ++name) {
        const unsigned char c = static_cast<unsigned char>(*name);
        out += (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '?';
    }
    return out;
}
std::string printable(const wchar_t* name) {
    std::string out;
    for (; name && *name && out.size() < 255; ++name)
        out += (*name >= 0x20 && *name < 0x7f) ? static_cast<char>(*name) : '?';
    return out;
}

void log_refused(const char* function, const std::string& name) {
    // Intentionally immortal: mapped-DLL teardown can run after normal statics are destroyed.
    static auto* mutex = new std::mutex;
    static auto* seen = new std::set<std::string>;
    const std::string key = std::string(function) + (name.empty() ? "" : " " + name);
    std::lock_guard lock(*mutex);
    if (seen->insert(key).second) std::fprintf(stderr, "[rt] dnsapi: offline, refused %s\n", key.c_str());
}

bool ascii_iequals(const wchar_t* a, const wchar_t* b) {
    for (; *a && *b; ++a, ++b) {
        const wchar_t ca = (*a >= L'A' && *a <= L'Z') ? *a + 32 : *a;
        const wchar_t cb = (*b >= L'A' && *b <= L'Z') ? *b + 32 : *b;
        if (ca != cb) return false;
    }
    return *a == L'\0' && *b == L'\0';
}

// A dotted IPv4 literal in plain decimal whose first octet is 127. A leading zero is octal to Windows (0127 is 87),
// so such a literal is not the loopback address it looks like and is refused.
bool is_ipv4_loopback_literal(const wchar_t* s) {
    unsigned first = 0;
    for (int part = 0; part < 4; ++part) {
        if (*s < L'0' || *s > L'9') return false;
        const wchar_t* start = s;
        unsigned value = 0;
        while (*s >= L'0' && *s <= L'9') {
            value = value * 10u + static_cast<unsigned>(*s - L'0');
            if (value > 255u) return false;
            ++s;
        }
        if (s - start > 1 && *start == L'0') return false;
        if (part == 0) first = value;
        if (part < 3) {
            if (*s != L'.') return false;
            ++s;
        }
    }
    return *s == L'\0' && first == 127u;
}

// ::1, or an IPv4-mapped loopback address (::ffff:127.x.y.z), parsed by ntdll as the host's DNS client does.
bool is_ipv6_loopback_literal(const wchar_t* s) {
    using ParseFn = LONG(NTAPI*)(PCWSTR, PCWSTR*, void*);
    static ParseFn parse = [] {
        const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
        return ntdll ? reinterpret_cast<ParseFn>(GetProcAddress(ntdll, "RtlIpv6StringToAddressW")) : nullptr;
    }();
    if (!parse) return false;
    uint8_t b[16] = {};
    PCWSTR end = nullptr;
    if (parse(s, &end, b) != 0 || !end || *end != L'\0') return false;
    for (int i = 0; i < 10; ++i)
        if (b[i] != 0) return false;
    if (b[10] == 0 && b[11] == 0) {  // ::1
        for (int i = 12; i < 15; ++i)
            if (b[i] != 0) return false;
        return b[15] == 1;
    }
    return b[10] == 0xff && b[11] == 0xff && b[12] == 127;  // ::ffff:127.0.0.0/104
}

// The names the host may answer: "localhost" and loopback literals. Nothing outside ASCII qualifies.
bool answers_locally(const wchar_t* name) {
    if (!name || !*name) return false;
    for (const wchar_t* p = name; *p; ++p)
        if (*p >= 0x80) return false;
    return ascii_iequals(name, L"localhost") || ascii_iequals(name, L"localhost.") ||
           is_ipv4_loopback_literal(name) || is_ipv6_loopback_literal(name);
}
bool answers_locally(const char* name) {
    if (!name) return false;
    std::wstring wide;
    for (const char* p = name; *p; ++p) {
        if (static_cast<unsigned char>(*p) >= 0x80) return false;
        wide += static_cast<wchar_t>(*p);
    }
    return answers_locally(wide.c_str());
}

DWORD local_options(ULONG64 options) { return (static_cast<DWORD>(options) & ~kNeedsNetwork) | kNoNetwork; }

template <typename Char>
using QueryFn = DNS_STATUS(WINAPI*)(const Char*, WORD, DWORD, PVOID, PDNS_RECORD*, PVOID*);

// The three DnsQuery_* entry points differ only in the name's encoding (A: the ANSI code page, UTF8, W). `host` is
// the host's function of the same name.
template <typename Char>
DNS_STATUS query(const char* function, QueryFn<Char> host, const Char* name, WORD type, DWORD options,
                 PDNS_RECORD* results, PVOID* reserved) {
    if (!name) return ERROR_INVALID_PARAMETER;  // measured: the host's answer for a null name
    if (!answers_locally(name)) {
        log_refused(function, *name ? printable(name) : "(empty)");
        if (results) *results = nullptr;
        return DNS_ERROR_RECORD_DOES_NOT_EXIST;
    }
    if (!host) return ERROR_PROC_NOT_FOUND;
    // pExtra can name DNS servers to ask, so it is not passed on.
    return host(name, type, local_options(options), nullptr, results, reserved);
}

DNS_STATUS refuse(const char* function) {
    log_refused(function, "");
    return ERROR_NETWORK_UNREACHABLE;
}

}  // namespace

DNS_STATUS WINAPI query_a(PCSTR name, WORD type, DWORD options, PVOID, PDNS_RECORD* results, PVOID* reserved) {
    static const QueryFn<char> host = host_proc<QueryFn<char>>("DnsQuery_A");
    return query("DnsQuery_A", host, name, type, options, results, reserved);
}
DNS_STATUS WINAPI query_utf8(PCSTR name, WORD type, DWORD options, PVOID, PDNS_RECORD* results, PVOID* reserved) {
    static const QueryFn<char> host = host_proc<QueryFn<char>>("DnsQuery_UTF8");
    return query("DnsQuery_UTF8", host, name, type, options, results, reserved);
}
DNS_STATUS WINAPI query_w(PCWSTR name, WORD type, DWORD options, PVOID, PDNS_RECORD* results, PVOID* reserved) {
    static const QueryFn<wchar_t> host = host_proc<QueryFn<wchar_t>>("DnsQuery_W");
    return query("DnsQuery_W", host, name, type, options, results, reserved);
}

DNS_STATUS WINAPI query_ex(PDNS_QUERY_REQUEST request, PDNS_QUERY_RESULT results, PDNS_QUERY_CANCEL) {
    // The host's own checks, measured: request version 1, 2 or 3, a result of version 1, a name.
    if (!request || !results || results->Version != DNS_QUERY_RESULTS_VERSION1 || !request->QueryName)
        return ERROR_INVALID_PARAMETER;
    if (request->Version != DNS_QUERY_REQUEST_VERSION1 && request->Version != DNS_QUERY_REQUEST_VERSION2 &&
        request->Version != DNS_QUERY_REQUEST_VERSION3)
        return ERROR_INVALID_PARAMETER;
    if (!answers_locally(request->QueryName)) {
        log_refused("DnsQueryEx", *request->QueryName ? printable(request->QueryName) : "(empty)");
        results->QueryStatus = DNS_ERROR_RECORD_DOES_NOT_EXIST;
        results->pQueryRecords = nullptr;
        return DNS_ERROR_RECORD_DOES_NOT_EXIST;
    }
    using QueryExFn = DNS_STATUS(WINAPI*)(PDNS_QUERY_REQUEST, PDNS_QUERY_RESULT, PDNS_QUERY_CANCEL);
    static const QueryExFn host = host_proc<QueryExFn>("DnsQueryEx");
    if (!host) return ERROR_PROC_NOT_FOUND;
    // A copy without servers or callback: the host answers at once, and its records go into the title's result.
    DNS_QUERY_REQUEST3 local{};
    if (request->Version == DNS_QUERY_REQUEST_VERSION3) {
        local = *reinterpret_cast<const DNS_QUERY_REQUEST3*>(request);
        local.IsNetworkQueryRequired = FALSE;
        local.cCustomServers = 0;
        local.pCustomServers = nullptr;
    } else {
        const DNS_QUERY_REQUEST& r = *request;
        local.Version = r.Version;
        local.QueryName = r.QueryName;
        local.QueryType = r.QueryType;
        local.QueryOptions = r.QueryOptions;
        local.InterfaceIndex = r.InterfaceIndex;
    }
    local.QueryOptions = local_options(local.QueryOptions) | (local.QueryOptions & ~ULONG64{0xffffffff});
    local.pDnsServerList = nullptr;
    local.pQueryCompletionCallback = nullptr;
    local.pQueryContext = nullptr;
    return host(reinterpret_cast<PDNS_QUERY_REQUEST>(&local), results, nullptr);
}

// Functions that reach a DNS server or the local link: refused before anything is sent. They take no parameters
// here, so nothing the title passed is read; on x64 the caller owns its argument area.
DNS_STATUS WINAPI query_raw() { return refuse("DnsQueryRaw"); }
DNS_STATUS WINAPI modify_records_a() { return refuse("DnsModifyRecordsInSet_A"); }
DNS_STATUS WINAPI modify_records_w() { return refuse("DnsModifyRecordsInSet_W"); }
DNS_STATUS WINAPI modify_records_utf8() { return refuse("DnsModifyRecordsInSet_UTF8"); }
DNS_STATUS WINAPI replace_record_set_a() { return refuse("DnsReplaceRecordSetA"); }
DNS_STATUS WINAPI replace_record_set_w() { return refuse("DnsReplaceRecordSetW"); }
DNS_STATUS WINAPI replace_record_set_utf8() { return refuse("DnsReplaceRecordSetUTF8"); }
DNS_STATUS WINAPI service_browse() { return refuse("DnsServiceBrowse"); }
DNS_STATUS WINAPI service_resolve() { return refuse("DnsServiceResolve"); }
DNS_STATUS WINAPI service_register() { return refuse("DnsServiceRegister"); }
DNS_STATUS WINAPI service_deregister() { return refuse("DnsServiceDeRegister"); }
DNS_STATUS WINAPI start_multicast_query() { return refuse("DnsStartMulticastQuery"); }

// The check succeeds and says the server could not be reached: it is never asked. A null server or status pointer
// fails as on the host (measured: a null server is ERROR_INVALID_PARAMETER).
DNS_STATUS WINAPI validate_server_status(PSOCKADDR server, PCWSTR, PDWORD status) {
    if (!server || !status) return ERROR_INVALID_PARAMETER;
    log_refused("DnsValidateServerStatus", "");
    *status = DNS_VALSVR_ERROR_UNREACHABLE;
    return ERROR_SUCCESS;
}

}  // namespace rt::dnsapi

namespace {
using namespace rt::dnsapi;
}  // namespace

SHIM_TABLE(dnsapi_offline,
    {"dnsapi.dll", "DnsQuery_A", reinterpret_cast<void*>(&query_a)},
    {"dnsapi.dll", "DnsQuery_UTF8", reinterpret_cast<void*>(&query_utf8)},
    {"dnsapi.dll", "DnsQuery_W", reinterpret_cast<void*>(&query_w)},
    {"dnsapi.dll", "DnsQueryEx", reinterpret_cast<void*>(&query_ex)},
    {"dnsapi.dll", "DnsQueryRaw", reinterpret_cast<void*>(&query_raw)},
    {"dnsapi.dll", "DnsModifyRecordsInSet_A", reinterpret_cast<void*>(&modify_records_a)},
    {"dnsapi.dll", "DnsModifyRecordsInSet_W", reinterpret_cast<void*>(&modify_records_w)},
    {"dnsapi.dll", "DnsModifyRecordsInSet_UTF8", reinterpret_cast<void*>(&modify_records_utf8)},
    {"dnsapi.dll", "DnsReplaceRecordSetA", reinterpret_cast<void*>(&replace_record_set_a)},
    {"dnsapi.dll", "DnsReplaceRecordSetW", reinterpret_cast<void*>(&replace_record_set_w)},
    {"dnsapi.dll", "DnsReplaceRecordSetUTF8", reinterpret_cast<void*>(&replace_record_set_utf8)},
    {"dnsapi.dll", "DnsServiceBrowse", reinterpret_cast<void*>(&service_browse)},
    {"dnsapi.dll", "DnsServiceResolve", reinterpret_cast<void*>(&service_resolve)},
    {"dnsapi.dll", "DnsServiceRegister", reinterpret_cast<void*>(&service_register)},
    {"dnsapi.dll", "DnsServiceDeRegister", reinterpret_cast<void*>(&service_deregister)},
    {"dnsapi.dll", "DnsStartMulticastQuery", reinterpret_cast<void*>(&start_multicast_query)},
    {"dnsapi.dll", "DnsValidateServerStatus", reinterpret_cast<void*>(&validate_server_status)},
);
