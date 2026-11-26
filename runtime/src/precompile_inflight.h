// gta6rt --precompile-pipelines: which compute entries are inside the driver right now, kept on disk so that a module
// that takes the whole process down (a driver crash, an abort in the compiler) is known on the next run.
//
// requests/inflight.txt lists, one per line, "<name> <key>" for the entries this process is building; an entry is
// added before its pipeline create and removed after it. The key names what was built: the module's bytes and the
// driver (the caller makes it), so a crash is held against that module on that driver only. The file is rewritten and
// closed at each change, so what it says survives the process (the OS keeps a closed file's data when the process
// dies; an OS crash is not what this is for).
// requests/crashed.txt collects what a dead run left in flight. A run starts by moving a leftover inflight.txt into
// crashed.txt. An entry is not built again while its current key is the one that crashed (it is reported); a new
// translation of the module or a new driver gives a new key, and the entry is tried again. The caller can forget every
// crash (--all).
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
        : inflight_(requests / "inflight.txt"), crashed_file_(requests / "crashed.txt") {
        std::error_code ec;
        std::filesystem::create_directories(requests, ec);
        crashed_ = read(crashed_file_);
        left_in_flight_ = read(inflight_);
        for (const auto& [name, key] : left_in_flight_) crashed_[name] = key;
        if (retry_crashed) crashed_.clear();
        write(crashed_file_, crashed_);
        std::filesystem::remove(inflight_, ec);
    }

    // What the run before this one left inside the driver (now also in crashed()).
    const Entries& left_in_flight() const { return left_in_flight_; }
    // Every entry a run died building, with the key it died on (until a build of it returns).
    const Entries& crashed() const { return crashed_; }
    // Whether `name`, as it is now (`key`), is what a run died building.
    bool is_crashed(const std::string& name, const std::string& key) const {
        const auto it = crashed_.find(name);
        return it != crashed_.end() && it->second == key;
    }

    // Around one entry's pipeline create, from any thread.
    void begin(const std::string& name, const std::string& key) {
        std::lock_guard lock(mutex_);
        active_[name] = key;
        write(inflight_, active_);
    }
    // The build returned: it is out of the driver, and a crash recorded for an earlier key of it no longer stands.
    void end(const std::string& name) {
        std::lock_guard lock(mutex_);
        active_.erase(name);
        write(inflight_, active_);
        if (crashed_.erase(name)) write(crashed_file_, crashed_);
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
    std::filesystem::path crashed_file_;
    Entries crashed_;
    Entries left_in_flight_;
    Entries active_;
    std::mutex mutex_;
};

}  // namespace leonida::precompile_inflight
