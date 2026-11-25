#pragma once

// Which registered buffer a console VA falls in (VulkanContext::buffer_at, for vertex and index views and
// WriteBufferImmediate). Ranges are stored in a self-balancing AVL interval tree keyed by (first VA, id) and
// augmented with per-subtree maximum inclusive end and maximum size. No Vulkan or Windows dependency; not
// thread-safe (VulkanContext guards it with resource_mutex_, like the bindings themselves).
//
// D3D12 lets placed buffers alias, so ranges may overlap. A lookup names the bytes it needs, [va, va + bytes), and
// returns, among the ranges that hold va, the one that starts closest below it (the larger ID on a tie) that also
// holds all of those bytes; when none does, the closest one that holds va. So a view that starts in a small placed
// buffer and runs past its end resolves to a larger buffer that aliases the same memory and holds all of it, instead
// of being cut at the small one's end (found in review of #2823).
//
// Because every node tracks the largest inclusive end in its subtree, find() prunes any subtree that cannot reach
// va (or va + bytes - 1), running in O(log N) worst-case time even on misses below or past a multi-gigabyte live
// buffer, and O(1) when va lies past every live range. Updating an existing (va, id) allocates nothing, and inserting
// a new range performs a single node allocation with strong exception safety.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace leonida {

template<typename Id>
class BufferVaIndex {
public:
    BufferVaIndex() = default;
    ~BufferVaIndex() { destroy(root_); }

    BufferVaIndex(const BufferVaIndex& other) : root_(clone(other.root_)), count_(other.count_) {}
    BufferVaIndex(BufferVaIndex&& other) noexcept
        : root_(std::exchange(other.root_, nullptr)), count_(std::exchange(other.count_, 0)) {}

    BufferVaIndex& operator=(const BufferVaIndex& other) {
        if (this == &other) return *this;
        BufferVaIndex copy(other);
        swap(copy);
        return *this;
    }
    BufferVaIndex& operator=(BufferVaIndex&& other) noexcept {
        if (this == &other) return *this;
        destroy(root_);
        root_ = std::exchange(other.root_, nullptr);
        count_ = std::exchange(other.count_, 0);
        return *this;
    }

    void swap(BufferVaIndex& other) noexcept {
        std::swap(root_, other.root_);
        std::swap(count_, other.count_);
    }

    // Registers [va, va + size). A null VA or an empty range is not indexed (no lookup can land in it).
    void insert(std::uint64_t va, std::uint64_t size, Id id) {
        if (!va || !size) return;
        bool added = false;
        root_ = insert_node(root_, va, size, id, added);
        if (added) ++count_;
    }

    // Forgets what insert(va, size, id) registered; nothing when it was not indexed.
    void erase(std::uint64_t va, Id id) {
        if (!root_) return;
        bool removed = false;
        root_ = erase_node(root_, va, id, removed);
        if (removed) --count_;
    }

    struct Hit {
        Id id;
        std::uint64_t start;  // the range's first VA
        std::uint64_t size;
    };

    // The range that holds `va`, preferring one that holds [va, va + bytes) (bytes 0 counts as 1); false when none
    // holds va.
    bool find(std::uint64_t va, Hit& out, std::uint64_t bytes = 1) const {
        if (!root_) return false;
        if (!bytes) bytes = 1;
        if (root_->max_end_inc < va) return false;
        if (bytes - 1 <= kMaxU64 - va) {
            const std::uint64_t need_inc = va + (bytes - 1);
            if (find_rightmost_fast(root_, va, bytes, need_inc, out)) return true;
            if (bytes == 1) return false;
            return find_rightmost_fast(root_, va, 1, va, out);
        }
        if (find_rightmost_wrap(root_, va, bytes, kMaxU64, out)) return true;
        return find_rightmost_fast(root_, va, 1, va, out);
    }

    // find() can return a shorter alias. Only a complete range can serve the view without new backing.
    bool covers(std::uint64_t va, std::uint64_t bytes) const {
        Hit hit{};
        return find(va, hit, bytes) && bytes <= hit.size - (va - hit.start);
    }

    std::size_t size() const { return count_; }
    std::uint64_t longest() const { return root_ ? root_->max_size : 0; }

private:
    static constexpr std::uint64_t kMaxU64 = std::numeric_limits<std::uint64_t>::max();

    struct Node {
        std::uint64_t start;
        Id id;
        std::uint64_t size;
        std::uint64_t max_end_inc;  // max over subtree of min(start + size - 1, UINT64_MAX)
        std::uint64_t max_size;     // max over subtree of size
        Node* left = nullptr;
        Node* right = nullptr;
        std::uint8_t height = 1;
    };

    static std::uint64_t end_inc(std::uint64_t start, std::uint64_t size) noexcept {
        const std::uint64_t last = size - 1;
        return last > kMaxU64 - start ? kMaxU64 : start + last;
    }

    static int key_cmp(std::uint64_t a_start, Id a_id, std::uint64_t b_start, Id b_id) noexcept {
        if (a_start != b_start) return a_start < b_start ? -1 : 1;
        if (a_id != b_id) return a_id < b_id ? -1 : 1;
        return 0;
    }

    static std::uint8_t node_height(const Node* n) noexcept { return n ? n->height : 0; }

    static int balance(const Node* n) noexcept {
        return int(node_height(n->left)) - int(node_height(n->right));
    }

    static void pull(Node* n) noexcept {
        std::uint64_t me = end_inc(n->start, n->size);
        std::uint64_t ms = n->size;
        std::uint8_t hl = 0, hr = 0;
        if (n->left) {
            if (n->left->max_end_inc > me) me = n->left->max_end_inc;
            if (n->left->max_size > ms) ms = n->left->max_size;
            hl = n->left->height;
        }
        if (n->right) {
            if (n->right->max_end_inc > me) me = n->right->max_end_inc;
            if (n->right->max_size > ms) ms = n->right->max_size;
            hr = n->right->height;
        }
        n->max_end_inc = me;
        n->max_size = ms;
        n->height = static_cast<std::uint8_t>(1 + (hl > hr ? hl : hr));
    }

    static Node* rotate_right(Node* y) noexcept {
        Node* x = y->left;
        y->left = x->right;
        x->right = y;
        pull(y);
        pull(x);
        return x;
    }

    static Node* rotate_left(Node* x) noexcept {
        Node* y = x->right;
        x->right = y->left;
        y->left = x;
        pull(x);
        pull(y);
        return y;
    }

    static Node* rebalance(Node* n) noexcept {
        pull(n);
        const int bf = balance(n);
        if (bf > 1) {
            if (balance(n->left) < 0) n->left = rotate_left(n->left);
            return rotate_right(n);
        }
        if (bf < -1) {
            if (balance(n->right) > 0) n->right = rotate_right(n->right);
            return rotate_left(n);
        }
        return n;
    }

    static Node* insert_node(Node* n, std::uint64_t va, std::uint64_t size, Id id, bool& added) {
        if (!n) {
            Node* created = new Node{va, id, size, end_inc(va, size), size, nullptr, nullptr, 1};
            added = true;
            return created;
        }
        const int cmp = key_cmp(va, id, n->start, n->id);
        if (cmp == 0) {
            n->size = size;
            pull(n);
            return n;
        }
        if (cmp < 0) n->left = insert_node(n->left, va, size, id, added);
        else n->right = insert_node(n->right, va, size, id, added);
        if (added) return rebalance(n);
        pull(n);
        return n;
    }

    static Node* extract_min(Node* n, Node*& min_out) noexcept {
        if (!n->left) {
            min_out = n;
            return n->right;
        }
        n->left = extract_min(n->left, min_out);
        return rebalance(n);
    }

    static Node* erase_node(Node* n, std::uint64_t va, Id id, bool& removed) noexcept {
        if (!n) return nullptr;
        const int cmp = key_cmp(va, id, n->start, n->id);
        if (cmp < 0) {
            n->left = erase_node(n->left, va, id, removed);
            return removed ? rebalance(n) : n;
        }
        if (cmp > 0) {
            n->right = erase_node(n->right, va, id, removed);
            return removed ? rebalance(n) : n;
        }
        removed = true;
        if (!n->left || !n->right) {
            Node* child = n->left ? n->left : n->right;
            delete n;
            return child;
        }
        Node* succ = nullptr;
        Node* right = extract_min(n->right, succ);
        succ->left = n->left;
        succ->right = right;
        delete n;
        return rebalance(succ);
    }

    // Iterative O(log N) search when target_end_inc = va + (bytes - 1) does not wrap UINT64_MAX:
    // for any subtree whose nodes all satisfy start <= va, max_end_inc >= target_end_inc guarantees a match exists,
    // so descending right -> self -> left never backtracks.
    static bool find_rightmost_fast(const Node* root, std::uint64_t va, std::uint64_t bytes,
                                    std::uint64_t target_end_inc, Hit& out) noexcept {
        // AVL height for <= 2^64 nodes is strictly below 80.
        const Node* ancestors[80];
        std::size_t depth = 0;
        const Node* cur = root;
        while (cur) {
            if (cur->max_end_inc < target_end_inc) break;
            if (cur->start > va) {
                cur = cur->left;
            } else {
                ancestors[depth++] = cur;
                cur = cur->right;
            }
        }
        while (depth) {
            const Node* n = ancestors[--depth];
            const std::uint64_t offset = va - n->start;
            if (offset < n->size && bytes <= n->size - offset) {
                out = {n->id, n->start, n->size};
                return true;
            }
            const Node* sub = n->left;
            if (!sub || sub->max_end_inc < target_end_inc) continue;
            // Every node in `sub` has start <= n->start <= va, and sub->max_end_inc >= target_end_inc guarantees a hit.
            while (sub) {
                if (sub->right && sub->right->max_end_inc >= target_end_inc) {
                    sub = sub->right;
                    continue;
                }
                const std::uint64_t sub_off = va - sub->start;
                if (sub_off < sub->size && bytes <= sub->size - sub_off) {
                    out = {sub->id, sub->start, sub->size};
                    return true;
                }
                sub = (sub->left && sub->left->max_end_inc >= target_end_inc) ? sub->left : nullptr;
            }
        }
        return false;
    }

    // Fallback when va + (bytes - 1) wraps past UINT64_MAX: max_end_inc saturates at UINT64_MAX so multiple wrapping
    // ranges may need inspection to verify exact (bytes <= size - offset).
    static bool find_rightmost_wrap(const Node* n, std::uint64_t va, std::uint64_t bytes,
                                    std::uint64_t target_end_inc, Hit& out) noexcept {
        if (!n || n->max_end_inc < target_end_inc) return false;
        if (n->start > va) return find_rightmost_wrap(n->left, va, bytes, target_end_inc, out);
        if (n->right && n->right->max_end_inc >= target_end_inc &&
            find_rightmost_wrap(n->right, va, bytes, target_end_inc, out))
            return true;
        const std::uint64_t offset = va - n->start;
        if (offset < n->size && bytes <= n->size - offset) {
            out = {n->id, n->start, n->size};
            return true;
        }
        return find_rightmost_wrap(n->left, va, bytes, target_end_inc, out);
    }

    static void destroy(Node* n) noexcept {
        if (!n) return;
        destroy(n->left);
        destroy(n->right);
        delete n;
    }

    static Node* clone(const Node* n) {
        if (!n) return nullptr;
        Node* copy = new Node{n->start, n->id, n->size, n->max_end_inc, n->max_size, nullptr, nullptr, n->height};
        try {
            copy->left = clone(n->left);
            copy->right = clone(n->right);
        } catch (...) {
            destroy(copy);
            throw;
        }
        return copy;
    }

    Node* root_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace leonida
