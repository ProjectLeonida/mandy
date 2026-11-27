// gta6rt --precompile-pipelines: which compute entries are inside the driver right now, kept on disk so that a module
// that takes the whole process down (a driver crash, an abort in the compiler) is known on the next run.
//
// requests/inflight.txt lists, one per line, "<name> <key>" for the entries this process is building; an entry is
// added before its pipeline create and removed after it. The key names what was built: the module file and the driver
// (the caller makes it). The file is rewritten and closed at each change, so what it says survives the process (the
// OS keeps a closed file's data when the process dies; an OS crash is not what this is for).
//
// A run that ends without a crash takes nothing to the next one: a stop it is told about (Ctrl+C, the console closing,
// the parent process gone) calls abandon(), which removes inflight.txt. A run killed from outside (TerminateProcess,
// a task manager) cannot, and leaves its list as a crash would, so one death proves nothing: an entry left in flight
// once becomes a suspect (requests/suspect.txt) and is built again; only an entry left in flight a second time with
// the same key, by the run that retried it, is held as crashed (requests/crashed.txt) and not built again. A driver
// crash on a module recurs on the retry; a kill lands on whatever happened to be building.
//
// A crash is held against that key only: a new translation of the module or a new driver gives a new key, the entry
// is tried again, and a build of it that returns clears it from both lists. The caller can forget every crash (--all).
//
// One precompile per journal folder: the caller holds its own lock (d3d12_vulkan_device.cpp) while this is in use, so
// a leftover inflight.txt can only be a dead run's. Standard library only.
#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <system_error>

namespace leonida::precompile_inflight {

class Record {
public:
    using Entries = std::map<std::string, std::string>;  // name -> key

    // `requests`: the journal's requests folder. `retry_crashed`: forget the earlier runs' crashes (--all).
    Record(std::filesystem::path requests, bool retry_crashed)
        : inflight_(requests / "inflight.txt"), suspect_file_(requests / "suspect.txt"),
          crashed_file_(requests / "crashed.txt") {
        std::error_code ec;
        std::filesystem::create_directories(requests, ec);
        crashed_ = read(crashed_file_);
        suspects_ = read(suspect_file_);
        left_in_flight_ = read(inflight_);
        for (const auto& [name, key] : left_in_flight_) {
            const auto suspect = suspects_.find(name);
            if (suspect != suspects_.end() && suspect->second == key) {
                crashed_[name] = key;  // a second death on the same module and driver
                suspects_.erase(suspect);
            } else {
                suspects_[name] = key;
            }
        }
        if (retry_crashed) {
            crashed_.clear();
            suspects_.clear();
        }
        write(crashed_file_, crashed_);
        write(suspect_file_, suspects_);
        std::filesystem::remove(inflight_, ec);
    }

    // What the run before this one left inside the driver (now a suspect, or crashed if it was one already).
    const Entries& left_in_flight() const { return left_in_flight_; }
    // Every entry held as crashed, with the key it died on (until a build of it returns).
    const Entries& crashed() const { return crashed_; }
    // Entries left in flight once: built again, and held as crashed if that run dies on them too.
    const Entries& suspects() const { return suspects_; }
    // Whether `name`, as it is now (`key`), is what runs died building twice.
    bool is_crashed(const std::string& name, const std::string& key) const {
        const auto it = crashed_.find(name);
        return it != crashed_.end() && it->second == key;
    }

    // Around one entry's pipeline create, from any thread.
    void begin(const std::string& name, const std::string& key) {
        std::lock_guard lock(mutex_);
        if (abandoned_) return;
        active_[name] = key;
        write(inflight_, active_);
    }
    // The build returned: it is out of the driver, and nothing recorded against it stands.
    void end(const std::string& name) {
        std::lock_guard lock(mutex_);
        if (abandoned_) return;
        active_.erase(name);
        write(inflight_, active_);
        if (crashed_.erase(name)) write(crashed_file_, crashed_);
        if (suspects_.erase(name)) write(suspect_file_, suspects_);
    }
    // The run is stopping for a reason that is not a crash: what is in flight now is not held against anything.
    // Safe from another thread (a console control handler); nothing is written after it.
    void abandon() {
        std::lock_guard lock(mutex_);
        abandoned_ = true;
        active_.clear();
        std::error_code ec;
        std::filesystem::remove(inflight_, ec);
    }

    const std::filesystem::path& crashed_path() const { return crashed_file_; }

private:
    // "<name> <key>" per line; a line with no key (an older file) has an empty one, which no build matches.
    static Entries read(const std::filesystem::path& path) {
        Entries entries;
        std::ifstream in(path, std::ios::binary);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            const auto space = line.rfind(' ');
            if (space == std::string::npos) entries[line] = std::string();
            else entries[line.substr(0, space)] = line.substr(space + 1);
        }
        return entries;
    }
    // Written beside and renamed over, so a reader never sees half a list; an empty list is no file.
    static void write(const std::filesystem::path& path, const Entries& entries) {
        std::error_code ec;
        if (entries.empty()) {
            std::filesystem::remove(path, ec);
            return;
        }
        auto next = path;
        next += ".new";
        const auto put = [&](const std::filesystem::path& to) {
            std::ofstream out(to, std::ios::binary | std::ios::trunc);
            for (const auto& [name, key] : entries) out << name << ' ' << key << '\n';
        };
        put(next);
        std::filesystem::rename(next, path, ec);
        if (ec) {  // a reader holding the old file open (Windows): write in place instead
            std::filesystem::remove(next, ec);
            put(path);
        }
    }

    std::filesystem::path inflight_;
    std::filesystem::path suspect_file_;
    std::filesystem::path crashed_file_;
    Entries crashed_;
    Entries suspects_;
    Entries left_in_flight_;
    Entries active_;
    std::mutex mutex_;
    bool abandoned_ = false;
};

}  // namespace leonida::precompile_inflight
