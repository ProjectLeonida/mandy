// Checks the offline DNS client shim (runtime/shims/dnsapi_offline.cpp).
//
// Resolves every shimmed export through rt::shims::find under dnsapi.dll and checks that the module stays host-owned
// (DnsRecordListFree, DnsFree and DnsCancelQuery are not shimmed). "localhost" and loopback literals are answered by
// the host with the network off, also when the title asks for the wire, a server list or a completion callback; every
// other name (.invalid, an empty name, other literals, an octal-looking 0127.0.0.1, *.localhost, non-ASCII) fails with
// DNS_ERROR_RECORD_DOES_NOT_EXIST and no records. The update, DNS-SD and multicast functions fail with
// ERROR_NETWORK_UNREACHABLE without calling back, and DnsValidateServerStatus reports the server unreachable. Only
// loopback names reach the host's DNS client, with DNS_QUERY_NO_WIRE_QUERY, so nothing touches the network.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <windns.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "shims.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <typename Fn>
Fn resolve(const char* sym) {
    return reinterpret_cast<Fn>(rt::shims::find("dnsapi.dll", sym));
}

using QueryWFn = DNS_STATUS(WINAPI*)(PCWSTR, WORD, DWORD, PVOID, PDNS_RECORD*, PVOID*);
using QueryAFn = DNS_STATUS(WINAPI*)(PCSTR, WORD, DWORD, PVOID, PDNS_RECORD*, PVOID*);
using QueryExFn = DNS_STATUS(WINAPI*)(PDNS_QUERY_REQUEST, PDNS_QUERY_RESULT, PDNS_QUERY_CANCEL);

PDNS_RECORD const kUntouched = reinterpret_cast<PDNS_RECORD>(static_cast<uintptr_t>(1));

// The single A record's address as text, or "" when the list is not exactly one A or AAAA record.
const char* only_address(PDNS_RECORD r, char (&buf)[64]) {
    buf[0] = '\0';
    if (!r || r->pNext) return buf;
    if (r->wType == DNS_TYPE_A) inet_ntop(AF_INET, &r->Data.A.IpAddress, buf, sizeof(buf));
    if (r->wType == DNS_TYPE_AAAA) inet_ntop(AF_INET6, &r->Data.AAAA.Ip6Address, buf, sizeof(buf));
    return buf;
}

// Queries `name` through the shim and expects one record with `address`; frees it with the host's function.
void expect_local(QueryWFn query, const wchar_t* name, WORD type, DWORD options, const char* address,
                  const char* what) {
    PDNS_RECORD records = kUntouched;
    const DNS_STATUS status = query(name, type, options, nullptr, &records, nullptr);
    char buf[64];
    const bool ok = status == ERROR_SUCCESS && records != kUntouched && std::strcmp(only_address(records, buf), address) == 0;
    if (!ok) std::printf("  status %ld, address '%s'\n", static_cast<long>(status), records != kUntouched ? buf : "-");
    check(ok, what);
    if (records && records != kUntouched) DnsRecordListFree(records, DnsFreeRecordList);
}

void expect_refused(QueryWFn query, const wchar_t* name, WORD type, const char* what) {
    PDNS_RECORD records = kUntouched;
    const DNS_STATUS status = query(name, type, DNS_QUERY_STANDARD, nullptr, &records, nullptr);
    if (status != DNS_ERROR_RECORD_DOES_NOT_EXIST || records != nullptr)
        std::printf("  status %ld\n", static_cast<long>(status));
    check(status == DNS_ERROR_RECORD_DOES_NOT_EXIST && records == nullptr, what);
    if (records && records != kUntouched) DnsRecordListFree(records, DnsFreeRecordList);
}

volatile LONG g_callbacks = 0;
void WINAPI count_callback(PVOID, PDNS_QUERY_RESULT result) {
    InterlockedIncrement(&g_callbacks);
    if (result && result->pQueryRecords) DnsRecordListFree(result->pQueryRecords, DnsFreeRecordList);
}

DNS_QUERY_REQUEST ex_request(PCWSTR name, WORD type, bool callback) {
    DNS_QUERY_REQUEST request{};
    request.Version = DNS_QUERY_REQUEST_VERSION1;
    request.QueryName = name;
    request.QueryType = type;
    request.QueryOptions = DNS_QUERY_STANDARD;
    request.pQueryCompletionCallback = callback ? count_callback : nullptr;
    return request;
}

}  // namespace

int main() {
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return 1;  // for inet_ntop only

    // ---- 1. Bindings ----
    const char* const shimmed[] = {"DnsQuery_A", "DnsQuery_UTF8", "DnsQuery_W", "DnsQueryEx", "DnsQueryRaw",
                                   "DnsModifyRecordsInSet_A", "DnsModifyRecordsInSet_W", "DnsModifyRecordsInSet_UTF8",
                                   "DnsReplaceRecordSetA", "DnsReplaceRecordSetW", "DnsReplaceRecordSetUTF8",
                                   "DnsServiceBrowse", "DnsServiceResolve", "DnsServiceRegister",
                                   "DnsServiceDeRegister", "DnsStartMulticastQuery", "DnsValidateServerStatus"};
    bool all = true;
    for (const char* name : shimmed) {
        if (!rt::shims::find("dnsapi.dll", name)) {
            std::printf("  not shimmed: %s\n", name);
            all = false;
        }
    }
    check(all, "every dnsapi.dll export that can send a packet is shimmed");
    check(!rt::shims::owns_dll("dnsapi.dll"), "dnsapi.dll stays host-owned (LoadLibrary returns the real module)");
    check(!rt::shims::find("dnsapi.dll", "DnsRecordListFree") && !rt::shims::find("dnsapi.dll", "DnsFree") &&
              !rt::shims::find("dnsapi.dll", "DnsCancelQuery") && !rt::shims::find("dnsapi.dll", "DnsQueryConfig"),
          "DnsRecordListFree, DnsFree, DnsCancelQuery and DnsQueryConfig stay the host's");

    const auto query_w = resolve<QueryWFn>("DnsQuery_W");
    const auto query_a = resolve<QueryAFn>("DnsQuery_A");
    const auto query_utf8 = resolve<QueryAFn>("DnsQuery_UTF8");
    const auto query_ex = resolve<QueryExFn>("DnsQueryEx");
    if (!query_w || !query_a || !query_utf8 || !query_ex) {
        std::printf("FAIL query entry points missing\n");
        return 1;
    }

    // ---- 2. Loopback names: the host answers them with the network off ----
    expect_local(query_w, L"localhost", DNS_TYPE_A, DNS_QUERY_STANDARD, "127.0.0.1", "DnsQuery_W localhost A is 127.0.0.1");
    expect_local(query_w, L"localhost", DNS_TYPE_AAAA, DNS_QUERY_STANDARD, "::1", "DnsQuery_W localhost AAAA is ::1");
    expect_local(query_w, L"LocalHost.", DNS_TYPE_A, DNS_QUERY_STANDARD, "127.0.0.1",
                 "a trailing dot and any case still name localhost");
    expect_local(query_w, L"127.0.0.1", DNS_TYPE_A, DNS_QUERY_STANDARD, "127.0.0.1", "the literal 127.0.0.1 is its own record");
    expect_local(query_w, L"127.45.6.7", DNS_TYPE_A, DNS_QUERY_STANDARD, "127.45.6.7", "any 127.0.0.0/8 literal is answered");
    expect_local(query_w, L"::1", DNS_TYPE_AAAA, DNS_QUERY_STANDARD, "::1", "the literal ::1 is its own record");
    expect_local(query_w, L"::ffff:127.0.0.1", DNS_TYPE_AAAA, DNS_QUERY_STANDARD, "::ffff:127.0.0.1",
                 "an IPv4-mapped loopback literal is answered");
    // The host refuses these next to DNS_QUERY_NO_WIRE_QUERY (ERROR_INVALID_PARAMETER, measured); the shim drops them.
    expect_local(query_w, L"localhost", DNS_TYPE_A, DNS_QUERY_WIRE_ONLY, "127.0.0.1", "DNS_QUERY_WIRE_ONLY does not reach the wire");
    expect_local(query_w, L"localhost", DNS_TYPE_A, DNS_QUERY_MULTICAST_ONLY, "127.0.0.1",
                 "DNS_QUERY_MULTICAST_ONLY does not reach the link");
    expect_local(query_w, L"localhost", DNS_TYPE_A, DNS_QUERY_BYPASS_CACHE, "127.0.0.1", "DNS_QUERY_BYPASS_CACHE is dropped");
    expect_local(query_w, L"localhost", DNS_TYPE_A, DNS_QUERY_RETURN_MESSAGE, "127.0.0.1", "DNS_QUERY_RETURN_MESSAGE is dropped");
    {
        // Other options reach the host unchanged: with DNS_QUERY_NO_LOCAL_NAME it refuses localhost, as it does itself.
        const DWORD off = DNS_QUERY_NO_WIRE_QUERY | DNS_QUERY_NO_NETBT | DNS_QUERY_NO_MULTICAST | DNS_QUERY_NO_HOSTS_FILE;
        PDNS_RECORD host_records = nullptr, shim_records = nullptr;
        const DNS_STATUS host = DnsQuery_W(L"localhost", DNS_TYPE_A, off | DNS_QUERY_NO_LOCAL_NAME, nullptr, &host_records, nullptr);
        const DNS_STATUS shim = query_w(L"localhost", DNS_TYPE_A, DNS_QUERY_NO_LOCAL_NAME, nullptr, &shim_records, nullptr);
        check(host != ERROR_SUCCESS && shim == host, "other options reach the host (DNS_QUERY_NO_LOCAL_NAME refuses localhost there too)");
        if (host_records) DnsRecordListFree(host_records, DnsFreeRecordList);
        if (shim_records) DnsRecordListFree(shim_records, DnsFreeRecordList);
    }
    {
        PDNS_RECORD records = kUntouched;
        DNS_STATUS status = query_a("localhost", DNS_TYPE_A, DNS_QUERY_STANDARD, nullptr, &records, nullptr);
        check(status == ERROR_SUCCESS && records && records != kUntouched && records->wType == DNS_TYPE_A &&
                  std::strcmp(reinterpret_cast<const char*>(records->pName), "localhost") == 0,
              "DnsQuery_A answers localhost with narrow record names");
        if (records && records != kUntouched) DnsRecordListFree(records, DnsFreeRecordList);
        records = kUntouched;
        status = query_utf8("localhost", DNS_TYPE_AAAA, DNS_QUERY_STANDARD, nullptr, &records, nullptr);
        check(status == ERROR_SUCCESS && records && records != kUntouched && records->wType == DNS_TYPE_AAAA,
              "DnsQuery_UTF8 answers localhost");
        if (records && records != kUntouched) DnsRecordListFree(records, DnsFreeRecordList);
    }

    // ---- 3. Every other name fails without records ----
    expect_refused(query_w, L"leonida-test.invalid", DNS_TYPE_A, "an external name fails with DNS_ERROR_RECORD_DOES_NOT_EXIST");
    expect_refused(query_w, L"leonida-test.invalid", DNS_TYPE_SRV, "an SRV lookup fails the same way");
    expect_refused(query_w, L"", DNS_TYPE_A, "an empty name (the host's own name) is refused");
    expect_refused(query_w, L"192.0.2.1", DNS_TYPE_A, "a literal outside loopback is refused");
    expect_refused(query_w, L"2001:db8::1", DNS_TYPE_AAAA, "an IPv6 literal outside loopback is refused");
    expect_refused(query_w, L"0127.0.0.1", DNS_TYPE_A, "0127.0.0.1 (octal to Windows) is refused");
    expect_refused(query_w, L"foo.localhost", DNS_TYPE_A, "a name under localhost is refused");
    expect_refused(query_w, L"localhost.example", DNS_TYPE_A, "a name that starts with localhost is refused");
    expect_refused(query_w, L"::1%1", DNS_TYPE_AAAA, "a scoped ::1 is refused");
    expect_refused(query_w, L"l\x043e" L"calhost", DNS_TYPE_A, "a name with a non-ASCII letter is refused");
    expect_refused(query_w, L"1.0.0.127.in-addr.arpa", DNS_TYPE_PTR, "a reverse lookup is refused");
    check(query_w(nullptr, DNS_TYPE_A, DNS_QUERY_STANDARD, nullptr, nullptr, nullptr) == ERROR_INVALID_PARAMETER,
          "a null name is ERROR_INVALID_PARAMETER, as on the host");
    check(query_w(L"leonida-test.invalid", DNS_TYPE_A, DNS_QUERY_STANDARD, nullptr, nullptr, nullptr) ==
              DNS_ERROR_RECORD_DOES_NOT_EXIST,
          "a refused name with no result pointer fails without writing");
    {
        PDNS_RECORD records = kUntouched;
        check(query_a("leonida-test.invalid", DNS_TYPE_A, DNS_QUERY_STANDARD, nullptr, &records, nullptr) ==
                      DNS_ERROR_RECORD_DOES_NOT_EXIST && records == nullptr,
              "DnsQuery_A refuses an external name");
        records = kUntouched;
        check(query_utf8("l\xc3\xb6" "calhost", DNS_TYPE_A, DNS_QUERY_STANDARD, nullptr, &records, nullptr) ==
                      DNS_ERROR_RECORD_DOES_NOT_EXIST && records == nullptr,
              "DnsQuery_UTF8 refuses a name with a non-ASCII letter");
    }

    // ---- 4. DnsQueryEx ----
    {
        DNS_QUERY_REQUEST request = ex_request(L"localhost", DNS_TYPE_A, false);
        DNS_QUERY_RESULT result{};
        result.Version = DNS_QUERY_RESULTS_VERSION1;
        char buf[64];
        DNS_STATUS status = query_ex(&request, &result, nullptr);
        check(status == ERROR_SUCCESS && result.QueryStatus == ERROR_SUCCESS &&
                  std::strcmp(only_address(result.pQueryRecords, buf), "127.0.0.1") == 0,
              "DnsQueryEx answers localhost");
        if (result.pQueryRecords) DnsRecordListFree(result.pQueryRecords, DnsFreeRecordList);

        // With a callback the host still answers localhost at once (measured), and the callback is not called.
        g_callbacks = 0;
        request = ex_request(L"localhost", DNS_TYPE_AAAA, true);
        result = {};
        result.Version = DNS_QUERY_RESULTS_VERSION1;
        DNS_QUERY_CANCEL cancel{};
        status = query_ex(&request, &result, &cancel);
        Sleep(100);
        check(status == ERROR_SUCCESS && std::strcmp(only_address(result.pQueryRecords, buf), "::1") == 0 &&
                  g_callbacks == 0,
              "DnsQueryEx with a callback completes localhost before it returns, without calling back");
        if (result.pQueryRecords) DnsRecordListFree(result.pQueryRecords, DnsFreeRecordList);

        request = ex_request(L"leonida-test.invalid", DNS_TYPE_A, true);
        result = {};
        result.Version = DNS_QUERY_RESULTS_VERSION1;
        result.pQueryRecords = kUntouched;
        status = query_ex(&request, &result, &cancel);
        Sleep(100);
        check(status == DNS_ERROR_RECORD_DOES_NOT_EXIST && result.QueryStatus == DNS_ERROR_RECORD_DOES_NOT_EXIST &&
                  result.pQueryRecords == nullptr && g_callbacks == 0,
              "DnsQueryEx refuses an external name at once, without calling back");

        // A server list (version 1) or custom servers (version 3) are dropped: localhost is still answered locally.
        DNS_ADDR_ARRAY servers{};
        servers.MaxCount = servers.AddrCount = 1;
        servers.Family = AF_INET;
        auto* server = reinterpret_cast<sockaddr_in*>(servers.AddrArray[0].MaxSa);
        server->sin_family = AF_INET;
        server->sin_port = htons(53);
        inet_pton(AF_INET, "192.0.2.53", &server->sin_addr);
        request = ex_request(L"localhost", DNS_TYPE_A, false);
        request.pDnsServerList = &servers;
        request.QueryOptions = DNS_QUERY_WIRE_ONLY;
        result = {};
        result.Version = DNS_QUERY_RESULTS_VERSION1;
        status = query_ex(&request, &result, nullptr);
        check(status == ERROR_SUCCESS && std::strcmp(only_address(result.pQueryRecords, buf), "127.0.0.1") == 0,
              "DnsQueryEx drops a server list and DNS_QUERY_WIRE_ONLY for localhost");
        if (result.pQueryRecords) DnsRecordListFree(result.pQueryRecords, DnsFreeRecordList);

        DNS_CUSTOM_SERVER custom{};
        custom.dwServerType = DNS_CUSTOM_SERVER_TYPE_UDP;
        reinterpret_cast<sockaddr_in*>(custom.MaxSa)->sin_family = AF_INET;
        reinterpret_cast<sockaddr_in*>(custom.MaxSa)->sin_port = htons(53);
        inet_pton(AF_INET, "192.0.2.53", &reinterpret_cast<sockaddr_in*>(custom.MaxSa)->sin_addr);
        DNS_QUERY_REQUEST3 request3{};
        request3.Version = DNS_QUERY_REQUEST_VERSION3;
        request3.QueryName = L"localhost";
        request3.QueryType = DNS_TYPE_A;
        request3.IsNetworkQueryRequired = TRUE;
        request3.cCustomServers = 1;
        request3.pCustomServers = &custom;
        result = {};
        result.Version = DNS_QUERY_RESULTS_VERSION1;
        status = query_ex(reinterpret_cast<PDNS_QUERY_REQUEST>(&request3), &result, nullptr);
        check(status == ERROR_SUCCESS && std::strcmp(only_address(result.pQueryRecords, buf), "127.0.0.1") == 0,
              "DnsQueryEx version 3 drops custom servers for localhost");
        if (result.pQueryRecords) DnsRecordListFree(result.pQueryRecords, DnsFreeRecordList);

        // The host's argument checks (measured), kept for refused names too.
        request = ex_request(L"leonida-test.invalid", DNS_TYPE_A, false);
        result = {};
        result.Version = DNS_QUERY_RESULTS_VERSION1;
        check(query_ex(nullptr, &result, nullptr) == ERROR_INVALID_PARAMETER, "DnsQueryEx: a null request is invalid");
        check(query_ex(&request, nullptr, nullptr) == ERROR_INVALID_PARAMETER, "DnsQueryEx: a null result is invalid");
        result.Version = 0;
        check(query_ex(&request, &result, nullptr) == ERROR_INVALID_PARAMETER, "DnsQueryEx: a result of version 0 is invalid");
        result.Version = DNS_QUERY_RESULTS_VERSION1;
        request.Version = 7;
        check(query_ex(&request, &result, nullptr) == ERROR_INVALID_PARAMETER, "DnsQueryEx: request version 7 is invalid");
        request.Version = DNS_QUERY_REQUEST_VERSION1;
        request.QueryName = nullptr;
        check(query_ex(&request, &result, nullptr) == ERROR_INVALID_PARAMETER, "DnsQueryEx: a null name is invalid");
    }

    // ---- 5. Updates, DNS-SD and multicast DNS are refused before anything is sent ----
    {
        using AnyFn = DNS_STATUS(WINAPI*)(void*, void*, void*, void*, void*, void*);
        const char* const senders[] = {"DnsQueryRaw", "DnsModifyRecordsInSet_A", "DnsModifyRecordsInSet_W",
                                       "DnsModifyRecordsInSet_UTF8", "DnsReplaceRecordSetA", "DnsReplaceRecordSetW",
                                       "DnsReplaceRecordSetUTF8", "DnsServiceDeRegister", "DnsStartMulticastQuery"};
        bool refused = true;
        for (const char* name : senders) {
            const DNS_STATUS status = resolve<AnyFn>(name)(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            if (status != ERROR_NETWORK_UNREACHABLE) {
                std::printf("  %s returned %ld\n", name, static_cast<long>(status));
                refused = false;
            }
        }
        check(refused, "raw queries, updates, DNS-SD deregistration and multicast queries fail with ERROR_NETWORK_UNREACHABLE");

        using BrowseFn = DNS_STATUS(WINAPI*)(PDNS_SERVICE_BROWSE_REQUEST, PDNS_SERVICE_CANCEL);
        using ResolveFn = DNS_STATUS(WINAPI*)(PDNS_SERVICE_RESOLVE_REQUEST, PDNS_SERVICE_CANCEL);
        using RegisterFn = DNS_STATUS(WINAPI*)(PDNS_SERVICE_REGISTER_REQUEST, PDNS_SERVICE_CANCEL);
        static volatile LONG service_callbacks = 0;
        DNS_SERVICE_BROWSE_REQUEST browse{};
        browse.Version = DNS_QUERY_REQUEST_VERSION1;
        browse.QueryName = L"_leonida-test._tcp.local";
        browse.pBrowseCallback = [](DWORD, PVOID, PDNS_RECORD records) {
            InterlockedIncrement(&service_callbacks);
            if (records) DnsRecordListFree(records, DnsFreeRecordList);
        };
        DNS_SERVICE_CANCEL cancel{};
        const DNS_STATUS browsed = resolve<BrowseFn>("DnsServiceBrowse")(&browse, &cancel);
        DNS_SERVICE_RESOLVE_REQUEST resolve_request{};
        resolve_request.Version = DNS_QUERY_REQUEST_VERSION1;
        resolve_request.QueryName = const_cast<PWSTR>(L"leonida-test._leonida-test._tcp.local");
        resolve_request.pResolveCompletionCallback = [](DWORD, PVOID, PDNS_SERVICE_INSTANCE instance) {
            InterlockedIncrement(&service_callbacks);
            if (instance) DnsServiceFreeInstance(instance);
        };
        const DNS_STATUS resolved = resolve<ResolveFn>("DnsServiceResolve")(&resolve_request, &cancel);
        DNS_SERVICE_REGISTER_REQUEST register_request{};
        register_request.Version = DNS_QUERY_REQUEST_VERSION1;
        register_request.pRegisterCompletionCallback = [](DWORD, PVOID, PDNS_SERVICE_INSTANCE instance) {
            InterlockedIncrement(&service_callbacks);
            if (instance) DnsServiceFreeInstance(instance);
        };
        const DNS_STATUS registered = resolve<RegisterFn>("DnsServiceRegister")(&register_request, &cancel);
        Sleep(100);
        check(browsed == ERROR_NETWORK_UNREACHABLE && resolved == ERROR_NETWORK_UNREACHABLE &&
                  registered == ERROR_NETWORK_UNREACHABLE && service_callbacks == 0,
              "DnsServiceBrowse, DnsServiceResolve and DnsServiceRegister fail at once and never call back");

        using ValidateFn = DNS_STATUS(WINAPI*)(PSOCKADDR, PCWSTR, PDWORD);
        const auto validate = resolve<ValidateFn>("DnsValidateServerStatus");
        sockaddr_in server{};
        server.sin_family = AF_INET;
        server.sin_port = htons(53);
        inet_pton(AF_INET, "192.0.2.53", &server.sin_addr);
        DWORD server_status = 0;
        check(validate(reinterpret_cast<PSOCKADDR>(&server), L"leonida-test.invalid", &server_status) == ERROR_SUCCESS &&
                  server_status == DNS_VALSVR_ERROR_UNREACHABLE,
              "DnsValidateServerStatus reports the server unreachable without asking it");
        check(validate(nullptr, L"x", &server_status) == ERROR_INVALID_PARAMETER &&
                  validate(reinterpret_cast<PSOCKADDR>(&server), L"x", nullptr) == ERROR_INVALID_PARAMETER,
              "DnsValidateServerStatus: a null server or status pointer is ERROR_INVALID_PARAMETER");
    }

    WSACleanup();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
