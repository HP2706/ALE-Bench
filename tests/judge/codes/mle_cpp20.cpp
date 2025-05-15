// https://atcoder.jp/contests/abc001/submissions/63920304
#include <vector>

int main() {
    std::vector<unsigned long long> v(128 * 1024 * 1024, 'a');  // Allocate 1 GiB of memory
    for (unsigned long long i = 0; i < (unsigned long long)v.size(); ++i) {
        v[i] = i;
    }
    return 0;
}
