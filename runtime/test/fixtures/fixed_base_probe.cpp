// A DLL with no relocations, linked /FIXED at one base: every copy of it loads at that address, so a test can load one
// copy, free it and get the next copy at the same HMODULE (library_loader_names_test). Built without the CRT; it does
// nothing and exports nothing. No game data.
#include <windows.h>

#ifdef __MSVC_RUNTIME_CHECKS
// Debug builds compile with /RTC1, whose checks call into the CRT; /NODEFAULTLIB leaves them unresolved.
extern "C" {
void __cdecl _RTC_InitBase() {}
void __cdecl _RTC_Shutdown() {}
}
#endif

extern "C" BOOL WINAPI fixed_base_probe_main(HINSTANCE, DWORD, LPVOID) { return TRUE; }
